#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace cv;
using namespace std;

vector<int> calcHistManuel(const Mat& kanal)
{
    CV_Assert(kanal.type() == CV_8UC1);

    vector<int> hist(256, 0);

    for (int y = 0; y < kanal.rows; y++)
    {
        const uchar* satir = kanal.ptr<uchar>(y);
        for (int x = 0; x < kanal.cols; x++)
        {
            hist[satir[x]]++;
        }
    }
    return hist;
}

Mat histCiz(const vector<int>& hist, Scalar renk, const string& baslik)
{
    const int genislik = 512, yukseklik = 400;
    const int binGenislik = genislik / 256;

    Mat cizim(yukseklik, genislik, CV_8UC3, Scalar(255, 255, 255));

    int maks = *max_element(hist.begin(), hist.end());
    if (maks == 0) maks = 1;

    for (int i = 0; i < 256; i++)
    {
        int h = cvRound((double)hist[i] / maks * (yukseklik - 40));
        rectangle(cizim,
            Point(i * binGenislik, yukseklik),
            Point((i + 1) * binGenislik - 1, yukseklik - h),
            renk, FILLED);
    }
    putText(cizim, baslik, Point(10, 25), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 0), 2);
    return cizim;
}

bool opencvIleKarsilastir(const Mat& kanal, const vector<int>& manuel)
{
    int histSize = 256;
    float aralik[] = { 0, 256 };
    const float* histAralik = { aralik };
    Mat ref;
    calcHist(&kanal, 1, 0, Mat(), ref, 1, &histSize, &histAralik);

    for (int i = 0; i < 256; i++)
        if (cvRound(ref.at<float>(i)) != manuel[i])
            return false;
    return true;
}

int main(int argc, char** argv)
{
    string yol = (argc > 1) ? argv[1] : "C:/Users/Kadir/Desktop/DIP-Digital-Image-Procressing/images/lenna.png";

    Mat resim = imread(yol, IMREAD_COLOR);
    if (resim.empty())
    {
        cerr << "Goruntu okunamadi: " << yol << endl;
        return -1;
    }

    vector<Mat> kanallar;
    split(resim, kanallar);   

    const char* isimler[3] = { "Mavi (B)", "Yesil (G)", "Kirmizi (R)" };
    const char* dosyalar[3] = { "hist_B.png", "hist_G.png", "hist_R.png" };
    Scalar renkler[3] = { Scalar(255, 0, 0), Scalar(0, 180, 0), Scalar(0, 0, 255) };

    for (int k = 0; k < 3; k++)
    {
        vector<int> hist = calcHistManuel(kanallar[k]);

        cout << isimler[k] << " kanal -> OpenCV ile ayni mi: "
            << (opencvIleKarsilastir(kanallar[k], hist) ? "evet" : "HAYIR") << endl;

        Mat cizim = histCiz(hist, renkler[k], string(isimler[k]) + " Kanal Histogrami");
        imshow(string(isimler[k]) + " Histogram", cizim);
        imwrite(dosyalar[k], cizim);
    }

    imshow("Orijinal", resim);
    waitKey(0);
    return 0;
}