#!/bin/bash
# Quick script to switch between Pokemon and CIFAR-10 tiles

POKEMON_DIR="pokemon_tiles"
CIFAR_DIR="cifar_tiles"
ACTIVE_DIR="tiles"

if [ "$1" == "pokemon" ]; then
    echo "Switching to Pokemon tiles..."
    rm -f $ACTIVE_DIR
    ln -s $POKEMON_DIR $ACTIVE_DIR
    echo "✓ Now using Pokemon tiles"
elif [ "$1" == "cifar" ]; then
    echo "Switching to CIFAR-10 tiles..."
    rm -f $ACTIVE_DIR
    ln -s $CIFAR_DIR $ACTIVE_DIR
    echo "✓ Now using CIFAR-10 tiles"
else
    echo "Usage: ./switch_tiles.sh [pokemon|cifar]"
    echo ""
    echo "Current tile counts:"
    echo "  Pokemon: $(ls $POKEMON_DIR 2>/dev/null | wc -l) tiles"
    echo "  CIFAR-10: $(ls $CIFAR_DIR 2>/dev/null | wc -l) tiles"
fi
