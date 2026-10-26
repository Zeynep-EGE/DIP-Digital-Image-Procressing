#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

Mat histHesapla(const Mat& kanal)
{
    int histSize = 256;
    float aralik[] = { 0, 256 };          
    const float* histAralik = { aralik };
    Mat hist;
    calcHist(&kanal, 1, 0, Mat(), hist, 1, &histSize, &histAralik);
    return hist;
}

Mat histCiz(const Mat& hist, Scalar renk, const string& baslik)
{
    const int genislik = 512, yukseklik = 400;
    const int binGenislik = cvRound((double)genislik / 256);

    Mat cizim(yukseklik, genislik, CV_8UC3, Scalar(255, 255, 255));

    Mat norm;
    normalize(hist, norm, 0, yukseklik - 30, NORM_MINMAX);

    for (int i = 1; i < 256; i++)
    {
        line(cizim,
            Point(binGenislik * (i - 1), yukseklik - cvRound(norm.at<float>(i - 1))),
            Point(binGenislik * i, yukseklik - cvRound(norm.at<float>(i))),
            renk, 2);
    }
    putText(cizim, baslik, Point(10, 25), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 0), 2);
    return cizim;
}

int main(int argc, char** argv)
{
    string yol = (argc > 1) ? argv[1] : "C:/Users/Kadir/Desktop/DIP-Digital-Image-Procressing/images/lenna.png";

    Mat resim = imread(yol, IMREAD_COLOR);
    if (resim.empty())
    {
        cerr << "Görüntü okunamadı: " << yol << endl;
        return -1;
    }

    vector<Mat> kanallar;
    split(resim, kanallar);  

    Mat histB = histHesapla(kanallar[0]);
    Mat histG = histHesapla(kanallar[1]);
    Mat histR = histHesapla(kanallar[2]);

    Mat cizimB = histCiz(histB, Scalar(255, 0, 0), "Mavi (B) Kanal Histogrami");
    Mat cizimG = histCiz(histG, Scalar(0, 180, 0), "Yesil (G) Kanal Histogrami");
    Mat cizimR = histCiz(histR, Scalar(0, 0, 255), "Kirmizi (R) Kanal Histogrami");

    imshow("Orijinal", resim);
    imshow("Mavi Histogram", cizimB);
    imshow("Yesil Histogram", cizimG);
    imshow("Kirmizi Histogram", cizimR);

    imwrite("hist_B.png", cizimB);
    imwrite("hist_G.png", cizimG);
    imwrite("hist_R.png", cizimR);

    waitKey(0);
    return 0;
}