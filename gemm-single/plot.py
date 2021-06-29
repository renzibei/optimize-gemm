import matplotlib.pyplot as plt

import re


N_list = [127, 128, 129, 255, 256, 257, 383, 384, 385, 511, 512, 513, 639, 640, 641, 767, 768, 769, 895, 896, 897, 1023, 1024, 1025, 1151, 1152, 1153, 1279, 1280, 1281]
final_list = [11.20, 16.40, 14.20, 13.50, 16.30, 15.30, 14.10, 16.10, 15.50, 14.60, 16.10, 15.60, 14.70, 16.00, 15.60, 14.90, 15.90, 15.60, 14.90, 15.80, 15.50, 14.90, 15.60, 15.50, 14.80, 15.50, 15.30, 14.70, 15.40, 15.2]
block_list = [10.90, 16.80, 13.10, 12.60, 13.80, 13.70, 13.10, 16.10, 14.20, 12.30, 11.90, 12.40, 13.60, 16.00, 14.10, 13.40, 14.20, 14.10, 13.70, 15.60, 14.20, 11.40, 11.70, 9.99, 13.70, 15.50, 14.10, 13.70, 14.40, 13.9]
simd_list = [10.60, 15.30, 13.00, 12.00, 12.80, 12.60, 12.20, 12.40, 13.40, 11.50, 5.80, 12.10, 10.70, 12.10, 11.20, 9.68, 5.20, 10.10, 9.13, 11.30, 9.28, 9.47, 3.84, 8.84, 8.76, 5.43, 8.86, 8.73, 3.91, 8.79]
square_reg_list = [5.75, 6.28, 6.44, 6.12, 5.52, 6.53, 6.29, 5.54, 6.32, 6.03, 5.04, 6.08, 5.91, 5.44, 5.75, 5.57, 4.31, 5.34, 5.40, 5.21, 5.06, 5.39, 3.64, 4.85, 5.19, 3.93, 4.93, 5.19, 3.65, 4.85]
square_list = [1.37, 1.34, 1.35, 1.32, 1.02, 1.32, 1.31, 0.98, 1.31, 1.31, 0.40, 1.31, 1.28, 1.00, 1.30, 1.30, 0.45, 1.30, 1.30, 1.03, 1.29, 1.30, 0.31, 1.29, 1.14, 0.61, 1.13, 0.96, 0.32, 0.967]
naive_list = [1.40, 1.38, 1.38, 1.34, 0.99, 1.34, 1.32, 1.00, 1.32, 1.32, 0.41, 1.32, 1.30, 1.00, 1.31, 1.30, 0.45, 1.30, 1.28, 1.02, 1.28, 1.27, 0.31, 1.28, 1.18, 0.68, 1.14, 1.00, 0.35, 0.986]
blas_list = [24.70, 31.00, 30.10, 29.10, 32.90, 32.10, 29.90, 32.60, 32.70, 31.40, 33.20, 33.20, 31.40, 33.00, 33.20, 31.90, 33.30, 33.30, 32.10, 33.40, 33.50, 32.30, 33.30, 33.40, 32.60, 33.50, 33.50, 32.60, 33.40, 33.5]


def parse_file():
    file_path = "./output.log"
    size_list = []
    gflop_list = []
    with open(file_path, "r") as in_f:
        lines = in_f.readlines()
        pattern = re.compile(r'Size:\s+([0-9]+)\s+Gflop/s:\s+([0-9.]+)\s+[^\)]+\)')
        for line in lines:
            m = pattern.match(line)
            if m:
                print(m.groups())
                size_num, gflop = m.groups()
                size_num = int(size_num)
                gflop = float(gflop)
                size_list.append(size_num)
                gflop_list.append(gflop)
    # print(size_list[0], gflop_list[0])
    count = len(size_list)
    for i in range(count - 1):
        print("%d, " % size_list[i], end='')
    print(size_list[-1])
    # print()
    for i in range(count - 1):
        print("%.2f, " % gflop_list[i], end='')
    print(gflop_list[-1])
    # plt.plot(size_list, gflop_list)
    # plt.show()

def show_plot():
    plt.plot(N_list, final_list, label="pack", marker='o')
    plt.plot(N_list, block_list, label="blocked", marker='v')
    plt.plot(N_list, simd_list, label="simd", marker='^')
    plt.plot(N_list, square_reg_list, label="4x4 register", marker='<')
    plt.plot(N_list, square_list, label="4x4 naive", marker='>')
    plt.plot(N_list, naive_list, label="naive", marker='p')
    # plt.plot(N_list, blas_list, label="blas", marker='*')
    plt.legend(loc="lower right")
    plt.ylabel("Gflop/s")
    plt.xlabel("Matrix Size N")
    plt.show()

def main():
    show_plot()
    # plt.plot(size_list, gflop_list)
    # plt.show()


if __name__ == '__main__':
    main()