#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdlib>

//bu kodları yapay zeka yardımı ile yazdım ama heyy kendimi geliştirmeye çalışıyorum

using namespace cv;
using namespace std;

Mat griyeCevirManuel(const Mat& bgr)
{
    CV_Assert(bgr.type() == CV_8UC3);

    Mat gri(bgr.rows, bgr.cols, CV_8UC1);

    for (int y = 0; y < bgr.rows; y++)
    {
        const Vec3b* kaynak = bgr.ptr<Vec3b>(y);
        uchar* hedef = gri.ptr<uchar>(y);

        for (int x = 0; x < bgr.cols; x++)
        {
            double b = kaynak[x][0];
            double g = kaynak[x][1];
            double r = kaynak[x][2];

            double deger = 0.299 * r + 0.587 * g + 0.114 * b;
            hedef[x] = saturate_cast<uchar>(cvRound(deger));
        }
    }
    return gri;
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

    Mat griOpenCV;
    cvtColor(resim, griOpenCV, COLOR_BGR2GRAY);

    Mat griManuel = griyeCevirManuel(resim);

    long long hataToplam = 0;
    long long farkliPikselSayisi = 0;
    int enBuyukFark = 0;

    for (int y = 0; y < griOpenCV.rows; y++)
    {
        const uchar* a = griOpenCV.ptr<uchar>(y);
        const uchar* b = griManuel.ptr<uchar>(y);

        for (int x = 0; x < griOpenCV.cols; x++)
        {
            int fark = abs((int)a[x] - (int)b[x]);
            hataToplam += fark;
            if (fark > 0) farkliPikselSayisi++;
            if (fark > enBuyukFark) enBuyukFark = fark;
        }
    }

    long long toplamPiksel = (long long)griOpenCV.rows * griOpenCV.cols;

    cout << "Goruntu boyutu            : " << resim.cols << " x " << resim.rows << endl;
    cout << "Toplam mutlak fark (hata) : " << hataToplam << endl;
    cout << "Piksel basina ortalama    : " << (double)hataToplam / toplamPiksel << endl;
    cout << "Farkli piksel sayisi      : " << farkliPikselSayisi << " / " << toplamPiksel << endl;
    cout << "En buyuk tekil fark       : " << enBuyukFark << endl;

    imshow("Orijinal", resim);
    imshow("Gri - OpenCV (cvtColor)", griOpenCV);
    imshow("Gri - Manuel", griManuel);
    imwrite("gri_opencv.png", griOpenCV);
    imwrite("gri_manuel.png", griManuel);

    waitKey(0);
    return 0;
}