# datathon2023-pictionary-challenge

# Contents
- [Pictionary Plunge About](#Pictionary-Plunge-About)
- [Install Google Cloud SDK](#Install-Google-Cloud-SDK)
- [Install Data](#Install-Data)
- [View Data](#View-Data)

## Pictionary Plunge About
The Pictionary Plunge is based off the Quick! Draw Challenge released by Google in 2016. A machine learning model will make classification of what the doodle is as the user makes the drawing.

## Install GCloud SDK
https://cloud.google.com/sdk/docs/install

## Install Data
gsutil -m cp 'gs://quickdraw_dataset/full/simplified/*.ndjson' .

## View Data
https://quickdraw.withgoogle.com/data

## Project Structure

### 1. Convert Stroke Maps to Image Map
Input drawings will initally be in the form of multidimensional arrays for each stroke, which is defined as a continous line where the pen does not lift from the drawing board. The values will include an x array and a y array.
