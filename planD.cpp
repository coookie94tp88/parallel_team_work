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
        for (int c_c = 0; c_c < m.cols * 3; c_c += 3) {
            b += data[m.step * r_c + c_c];
            g += data[m.step * r_c + c_c + 1];
            r += data[m.step * r_c + c_c + 2];
        }
    }
    unsigned nPix = static_cast<unsigned>(m.rows) * static_cast<unsigned>(m.cols);

    return Vec3b(
        static_cast<unsigned char>(b / nPix),
        static_cast<unsigned char>(g / nPix),
        static_cast<unsigned char>(r / nPix)
    );
}

/*
self explanatory
*/
void printProgress(int percentage, unsigned elapsed, unsigned etl) {
    cout << "\rProgress:|";
    const int bar_length = 15;
    int number_of_progress_chars = static_cast<int>(round(percentage * bar_length / 100.0));

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
    case 3: //rotate 180
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
void computePixelsAndIndex(const string& images_root,
    const string& out_folder,
    Size s,
    const string& index_filename) {

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

        clock_t end = clock();
        int percentage = static_cast<int>(round(double(i) * 100.0 / images.size()));

        unsigned elapsed = static_cast<unsigned>((end - begin) / CLOCKS_PER_SEC);
        unsigned etl = (i == 0)
            ? 0u
            : static_cast<unsigned>((double(elapsed) / i) * (images.size() - i));

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

        im.mean = Vec3b(
            static_cast<unsigned char>(b),
            static_cast<unsigned char>(g),
            static_cast<unsigned char>(r)
        );
        index.push_back(im);
    }
}

/*
returns the nearest image given the pixel color and the forbidden ones
Plan A：不做 sort，而是線性掃描 + 保留目前最佳（用平方距離，避免 sqrt）
*/
ImageMean nearestImage(const vector<ImageMean>& index,
    const Vec3b& color,
    vector<unsigned char>& forbidden) {
    double bestDist = std::numeric_limits<double>::max();
    int bestIdx = -1;

    for (size_t i = 0; i < index.size(); ++i) {
        if (forbidden[i]) continue; // 跳過最近用過的 tiles

        const Vec3b& conf = index[i].mean;
        int db = int(color[0]) - int(conf[0]);
        int dg = int(color[1]) - int(conf[1]);
        int dr = int(color[2]) - int(conf[2]);

        // squared distance
        double dist = double(db * db + dg * dg + dr * dr);

        if (dist < bestDist) {
            bestDist = dist;
            bestIdx = static_cast<int>(i);

            // 如果完全相同（距離 0），可以提早結束
            if (dist == 0.0) break;
        }
    }

    ImageMean result;
    if (bestIdx >= 0) {
        forbidden[static_cast<size_t>(bestIdx)] = 1;
        result = index[static_cast<size_t>(bestIdx)];
    }
    return result;
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
            if (!line.empty() && line[0] != '#')
                strings.push_back(line);
        }

        if (strings.size() < 8) {
            throw 2;
        }

        istringstream ss;
        // dataset
        params.images_root = strings[0];
        // pixel_folder
        params.out_folder = strings[1];
        // pixel_size
        ss = istringstream(strings[2]);
        unsigned sx, sy;
        ss >> sx >> sy;
        params.pixelSize = Size(static_cast<int>(sx), static_cast<int>(sy));
        // input image
        params.input_image = strings[3];
        // resize
        ss = istringstream(strings[4]);
        ss >> params.resize_x >> params.resize_y;
        // pixelize
        ss = istringstream(strings[5]);
        params.pixelize = static_cast<char>(atoi(ss.str().c_str()));
        // mosaicize
        ss = istringstream(strings[6]);
        params.mosaicize = static_cast<char>(atoi(ss.str().c_str()));
        // skip_interval
        ss = istringstream(strings[7]);
        ss >> params.skip_interval;
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

        Mat3b output(src.rows * p.pixelSize.height,
            src.cols * p.pixelSize.width);

        vector<unsigned char> forbidden(index.size(), 0);

        for (int i = 0; i < src.rows; ++i) {
            for (int j = 0; j < src.cols; ++j) {

                if (((i * src.cols + j) % p.skip_interval) == 0)
                    forbidden.assign(index.size(), 0);

                Vec3b color = src(i, j);

                ImageMean best_match = nearestImage(index, color, forbidden);
                Mat3b pixel = imread(best_match.name);
                if (pixel.empty()) {
                    continue;
                }

                Rect bound(j * p.pixelSize.width,
                    i * p.pixelSize.height,
                    p.pixelSize.width,
                    p.pixelSize.height);

                pixel.copyTo(
                    output.rowRange(i * p.pixelSize.height,
                        (i + 1) * p.pixelSize.height)
                    .colRange(j * p.pixelSize.width,
                        (j + 1) * p.pixelSize.width)
                );
            }
        }

        imwrite("outputD.png", output);
        cout << "Done. Mosaic image has been written to output.png" << endl;
    }

    return 0;
}