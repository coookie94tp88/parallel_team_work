#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <boost/filesystem.hpp>
#include <iomanip>
#ifdef _WIN32
#include <direct.h>   // Windows: _mkdir, _getcwd, etc.
#else
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>    // for getcwd, chdir on Unix-like
// map common Windows names to POSIX ones if code uses _mkdir/_getcwd/_chdir
#ifndef _mkdir
#define _mkdir(dir) mkdir((dir), 0755)
#endif
#ifndef _getcwd
#define _getcwd(buf, size) getcwd((buf), (size))
#endif
#ifndef _chdir
#define _chdir(dir) chdir(dir)
#endif
#endif
#include <ctime>
#include "exif.h"

namespace fs = ::boost::filesystem;

using namespace std;
using namespace cv;

/*
structure to map an image to its mean color
*/
struct ImageMean {
    string name;
    Vec3b mean;
};

/*
structure to hold application parameters
*/
struct Params {
    string images_root;
    string out_folder;
    string index_filename = "index.txt";
    Size pixelSize;
    string input_image;
    double resize_x;
    double resize_y;
    char pixelize;
    char mosaicize;
    unsigned skip_interval;

    // ==== 新增：亮度 / 顏色調整相關參數 ====
    int  brightness_adjust = 0;    // 0 = off, 1 = on
    int  brightness_max_delta = 30;   // 最大亮度補償 (灰階值)
    double color_adjust_strength = 1.0; // 0.0 ~ 1.0
};

/*
utility: check string in vector
*/
bool isInVector(const string& str, const vector<string>& vec) {
    for (const string& s : vec) {
        if (s == str)
            return true;
    }
    return false;
}

/*
seeks recursively in root for all files having extension ext,
and builds the list ret
*/
void get_all(const fs::path& root, const vector<string>& extensions, vector<string>& ret)
{
    if (!fs::exists(root) || !fs::is_directory(root)) return;

    fs::recursive_directory_iterator it(root);
    fs::recursive_directory_iterator endit;

    while (it != endit)
    {
        if (fs::is_regular_file(*it) && isInVector(it->path().extension().string(), extensions))
            ret.push_back(it->path().string());
        ++it;
    }
}

/*
computes the mean color of an image
*/
Vec3b meanColor(const Mat3b& m) {
    unsigned long b = 0;
    unsigned long g = 0;
    unsigned long r = 0;

    const unsigned char* data = (unsigned char*)(m.data);
    for (int r_c = 0; r_c < m.rows; ++r_c) {
        for (int c_c = 0; c_c < m.cols * 3; c_c = c_c + 3) {
            b += data[m.step * r_c + c_c];
            g += data[m.step * r_c + c_c + 1];
            r += data[m.step * r_c + c_c + 2];
        }
    }
    unsigned nPix = static_cast<unsigned>(m.rows) * static_cast<unsigned>(m.cols);

    return Vec3b(static_cast<unsigned char>(b / nPix),
        static_cast<unsigned char>(g / nPix),
        static_cast<unsigned char>(r / nPix));
}

/*
computes the mean brightness (grayscale) of an image
用來做亮度調整
*/
double meanBrightness(const Mat3b& m) {
    // 使用 OpenCV 的 mean + BGR 灰階權重
    // gray = 0.114*B + 0.587*G + 0.299*R
    Scalar mBGR = cv::mean(m);
    double B = mBGR[0];
    double G = mBGR[1];
    double R = mBGR[2];
    return 0.114 * B + 0.587 * G + 0.299 * R;
}

/*
self explanatory
*/
void printProgress(int percentage, unsigned elapsed, unsigned etl) {
    cout << "\rProgress:|";
    char bar_length = 15;
    char number_of_progress_chars = static_cast<char>(round(percentage * bar_length / 100.0));

    for (int j = 0; j < number_of_progress_chars; ++j) cout << "=";
    cout << ">";
    for (int j = 0; j < bar_length - number_of_progress_chars; ++j) cout << " ";
    cout << "| " << percentage << "%, Time elapsed: " << elapsed << " seconds, ETL: " << etl << " seconds.";
}

/*
extracts exif orientations from jpeg files.
useful if you have pictures taken with smartphones
*/
char extractEXIFOrientation(const string& img_name) {
    FILE* fp = fopen(img_name.c_str(), "rb");
    if (!fp) {
        return 1; // assume normal
    }

    fseek(fp, 0, SEEK_END);
    unsigned long fsize = ftell(fp);
    rewind(fp);
    unsigned char* buf = new unsigned char[fsize];
    if (fread(buf, 1, fsize, fp) != fsize) {
        delete[] buf;
        fclose(fp);
        return 1;
    }
    fclose(fp);

    // Parse EXIF
    easyexif::EXIFInfo result;
    int code = result.parseFrom(buf, fsize);
    delete[] buf;
    if (code) {
        return 1;
    }

    return result.Orientation;
}

/*
rotates an image clockwise
*/
void rotateClockwise(Mat& img) {
    transpose(img, img);
    flip(img, img, 1);
}

/*
fixes image given its exif orientation
*/
void rectifyByOrientation(Mat3b& img, char orientation) {
    switch (orientation) {
    case 1: break; //fine
    case 6: //rotate clockwise
        rotateClockwise(img);
        break;
    case 3: //flip vertically (rotate 180)
        rotateClockwise(img);
        rotateClockwise(img);
        break;
    case 8: // rotate counterclockwise
        rotateClockwise(img);
        rotateClockwise(img);
        rotateClockwise(img);
        break;
    default: break;
    }
}

/*
precomputes all the small images ("pixels") that will form the mosaic.
Also stores in an index (index_filename) the mapping with the mean color.
*/
void
computePixelsAndIndex(const string& images_root, const string& out_folder, Size s, const string& index_filename) {

    cout << "Pixelizing images from " << images_root << ". This might take a while." << endl;
    vector<string> images;
    vector<string> extensions = { ".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG" };
    get_all(images_root, extensions, images);

    fs::remove_all(out_folder);
    fs::create_directory(out_folder);
    ofstream out_index(out_folder + index_filename);

    clock_t begin = clock();

    for (unsigned i = 0; i < images.size(); ++i)
    {
        Mat3b img = imread(images[i]);
        if (img.empty()) {
            continue;
        }

        char orientation = extractEXIFOrientation(images[i]);
        rectifyByOrientation(img, orientation);

        Vec3b mean_color = meanColor(img);

        Mat3b resized;
        resize(img, resized, s);

        ostringstream out_filename;
        out_filename << out_folder << setfill('0') << setw(5) << i << ".png";
        imwrite(out_filename.str(), resized);

        out_index << out_filename.str() << "\t"
            << static_cast<int>(mean_color[0]) << " "
            << static_cast<int>(mean_color[1]) << " "
            << static_cast<int>(mean_color[2]) << endl;

        //Measuring time for progress...
        clock_t end = clock();
        int percentage = static_cast<int>(round(double(i) * 100.0 / images.size()));

        unsigned elapsed = static_cast<unsigned>(double(end - begin) / CLOCKS_PER_SEC);
        unsigned etl = (i == 0) ? 0u : static_cast<unsigned>((double(elapsed) / i) * (images.size() - i));

        printProgress(percentage, elapsed, etl);
    }
    out_index.close();

    cout << endl << "Done." << endl;
}

/*
reads the precomputed mean colors from file
*/
void readIndexFile(const string& index_filename, vector<ImageMean>& index) {
    ifstream in(index_filename);
    string line;
    while (getline(in, line)) {
        stringstream ss(line);
        ImageMean im;
        unsigned r, g, b;
        ss >> im.name >> b >> g >> r;

        im.mean = Vec3b(static_cast<unsigned char>(b),
            static_cast<unsigned char>(g),
            static_cast<unsigned char>(r));
        index.push_back(im);
    }
}

/*
utility structure for sorting (by similarity)
*/
struct idxVal {
    int idx;
    double val;

    bool operator<(idxVal conf) const {
        return val < conf.val;
    }
};

/*
returns the nearest image given the pixel color and the forbidden ones
*/
ImageMean nearestImage(const vector<ImageMean>& index, const Vec3b& color, vector<unsigned char>& forbidden) {
    vector<idxVal> ivals;
    ivals.reserve(index.size());

    for (size_t i = 0; i < index.size(); ++i) {
        const Vec3b& conf = index[i].mean;
        idxVal ival;
        ival.idx = static_cast<int>(i);
        // 使用平方距離，避免 sqrt 的開銷
        int db = int(color[0]) - int(conf[0]);
        int dg = int(color[1]) - int(conf[1]);
        int dr = int(color[2]) - int(conf[2]);
        ival.val = double(db * db + dg * dg + dr * dr);
        ivals.push_back(ival);
    }
    sort(ivals.begin(), ivals.end());

    for (size_t i = 0; i < ivals.size(); i++) {
        int idx = ivals[i].idx;
        if (!forbidden[static_cast<size_t>(idx)]) {
            forbidden[static_cast<size_t>(idx)] = 1;
            return index[static_cast<size_t>(idx)];
        }
    }
    ImageMean err;
    return err;
}

/*
self explanatory
*/
void printUsage() {
    cout << "Usage: Photomosaic <settings_file.ini>" << endl;
}

/*
reads the file holding parameters settings
*/
void readInitFile(const string& init_file, Params& params)
{
    try {
        ifstream in(init_file);
        if (!in.is_open()) {
            throw 1;
        }

        string line;
        vector<string> strings;
        while (getline(in, line)) {
            if (!line.empty() && line[0] != '#') strings.push_back(line);
        }
        istringstream ss;
        //dataset
        params.images_root = strings[0];
        //pixel_folder
        params.out_folder = strings[1];
        //pixel_size
        ss = istringstream(strings[2]);
        unsigned sx, sy;
        ss >> sx >> sy;
        params.pixelSize = Size(static_cast<int>(sx), static_cast<int>(sy));
        //input image
        params.input_image = strings[3];
        //resize
        ss = istringstream(strings[4]);
        ss >> params.resize_x >> params.resize_y;
        //pixelize
        ss = istringstream(strings[5]);
        params.pixelize = static_cast<char>(atoi(ss.str().c_str()));
        //mosaicize
        ss = istringstream(strings[6]);
        params.mosaicize = static_cast<char>(atoi(ss.str().c_str()));
        //skip_interval
        ss = istringstream(strings[7]);
        ss >> params.skip_interval;

        // ==== 新增：若 ini 裡有更多設定就讀進來（否則用 default） ====
        // brightness_adjust
        if (strings.size() > 8) {
            ss = istringstream(strings[8]);
            ss >> params.brightness_adjust;
        }
        // brightness_max_delta
        if (strings.size() > 9) {
            ss = istringstream(strings[9]);
            ss >> params.brightness_max_delta;
        }
        // color_adjust_strength
        if (strings.size() > 10) {
            ss = istringstream(strings[10]);
            ss >> params.color_adjust_strength;
        }
    }
    catch (...) {
        cout << "Something went wrong in the initialization file parsing... please check it." << endl;
        exit(1);
    }
}

/*
main function
*/
int main(int argc, char** argv) {

    if (argc != 2) {
        printUsage();
        return 1;
    }

    string init_file = argv[1];
    Params p;
    readInitFile(init_file, p);

    if (p.pixelize) {
        computePixelsAndIndex(p.images_root, p.out_folder, p.pixelSize, p.index_filename);
    }

    if (p.mosaicize) {
        cout << "Rendering mosaic for image " << p.input_image << "..." << endl;
        vector<ImageMean> index;
        readIndexFile(p.out_folder + p.index_filename, index);

        if (index.empty()) {
            cout << "Index is empty. Did you run pixelization?" << endl;
            return 1;
        }

        Mat3b src = imread(p.input_image);
        if (src.empty()) {
            cout << "Cannot read input image: " << p.input_image << endl;
            return 1;
        }

        resize(src, src, Size(0, 0), p.resize_x, p.resize_y);

        Mat3b output(src.rows * p.pixelSize.height, src.cols * p.pixelSize.width);
        vector<unsigned char> forbidden(index.size());

        clock_t begin = clock();
        for (int i = 0; i < src.rows; ++i) {
            for (int j = 0; j < src.cols; ++j) {

                if (((i * src.cols + j) % p.skip_interval) == 0)
                    forbidden = vector<unsigned char>(index.size(), 0);

                Vec3b color = src(i, j);

                ImageMean best_match = nearestImage(index, color, forbidden);
                Mat3b pixel = imread(best_match.name);
                if (pixel.empty()) {
                    continue;
                }

                // ==== 亮度 / 顏色調整 C 方案 ====
                if (p.brightness_adjust) {
                    // 目標像素亮度 (單一像素，用同樣灰階權重)
                    double targetL = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2];

                    // tile 平均亮度
                    double tileMeanL = meanBrightness(pixel);

                    double rawDelta = targetL - tileMeanL;

                    // 限制最大補償幅度
                    double maxD = static_cast<double>(p.brightness_max_delta);
                    if (rawDelta > maxD)  rawDelta = maxD;
                    if (rawDelta < -maxD) rawDelta = -maxD;

                    // 加入強度參數
                    double delta = rawDelta * p.color_adjust_strength;

                    // 套用到整張 tile
                    cv::Mat pixelFloat;
                    pixel.convertTo(pixelFloat, CV_32FC3);
                    pixelFloat += cv::Scalar(delta, delta, delta);

                    // clip 到 [0,255]
                    cv::Mat pixelClipped;
                    cv::min(pixelFloat, 255.0, pixelFloat);
                    cv::max(pixelFloat, 0.0, pixelFloat);
                    pixelFloat.convertTo(pixelClipped, CV_8UC3);

                    pixel = pixelClipped;
                }

                Rect bound(j * p.pixelSize.width,
                    i * p.pixelSize.height,
                    p.pixelSize.width,
                    p.pixelSize.height);

                pixel.copyTo(
                    output.rowRange(i * p.pixelSize.height, i * p.pixelSize.height + p.pixelSize.height)
                    .colRange(j * p.pixelSize.width, j * p.pixelSize.width + p.pixelSize.width)
                );
            }
        }

        imwrite("outputB.png", output);

        cout << endl << "Done. Mosaic image has been written to output.png" << endl;
    }

    return 0;
}