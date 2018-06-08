#!/usr/bin/env bash
ffmpeg -c:v png -r 5 -i new/%03d.png -q:a 0 -q:v 0 -c:v wmv2 -c:a wmav2 -r 30 video.wmv
