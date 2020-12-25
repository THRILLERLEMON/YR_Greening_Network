# -*-coding:utf-8*-
import sys
import os
import xlrd
import xlwt
import xlsxwriter
import openpyxl
import pandas as pd
import numpy as np
import os

def push_excel_information(excel_path):
    # 打开文件
    work_book = xlrd.open_workbook(excel_path)
    # 获取工作簿中所有sheet表对象
    sheets = work_book.sheets()
    # print(sheets)
    # 获取工作簿所有sheet表对象名称
    sheets_name = work_book.sheet_names()
    return work_book, work_book.sheets, sheets, sheets_name

def excel_interpolation_start_end(start_index, end_index, value_index_left, value_index_right, inter_list):
    # 01对于首尾都没有数据的部分进行插补
    for i in range(start_index, value_index_left):
        inter_list[i] = inter_list[value_index_left]
        i = i + 1
    for i in range(value_index_right + 1, end_index):
        inter_list[i] = inter_list[value_index_right]
        i = i + 1
    return inter_list

def excel_interpolation_middle(x, y, order):
    # 02对于中间部分进行插补,多项式
    # start_index: 线性插值起始位置
    # end_index: 线性插值结束位置
    # num: 缺失个数
    # inter_list: 待插值列表
    # interp_index_nan: 空值索引
    # x, y, order: 多项式拟合 x，y，阶数
    x_new_value = []
    x_new_nan = []
    y_new_value = []
    index1 = [i for i ,value in enumerate(y) if value != '']
    index2 = [i for i ,value in enumerate(y) if value == '']
    for j in index1:
        y_new_value.append(y[j])
        x_new_value.append(x[j])
    if index2 == []:
        x_new_nan = index1
    else:
        for m in index2:
            x_new_nan.append(x[m])
    a = np.polyfit(x_new_value,y_new_value,order)#用2次多项式拟合x，y数组
    b = np.poly1d(a)#拟合完之后用这个函数来生成多项式对象
    c = b(x_new_nan)#生成多项式对象之后，就是获取x在这个多项式处的值
    k = 0
    for n in index2:
        y[n] = c[k]
        k = k + 1
    return y

def interpolation2excel(list, index_name, excel_temp, col_name):
    dic = dict(zip(col_name, list))
    df = pd.DataFrame(dic)
    df.to_excel(excel_temp, sheet_name = index_name, columns = col_name)
    return

def excel_interpolation(work_book, sheet_name):
    # 插补主函数
    for index_name in sheet_name:
        df = pd.DataFrame()
        df.to_excel('F:/Learning/Yanjiusheng1/复杂网络/统计数据插补/%s.xlsx' %index_name)
        excel_temp = 'F:/Learning/Yanjiusheng1/复杂网络/统计数据插补/%s.xlsx' %index_name

        sheet_temp = work_book.sheet_by_name(index_name)
        # 获取sheet表对象有效行数 列数
        row_sum = sheet_temp.nrows
        col_sum = sheet_temp.ncols
        col_start = 14
        x =[ i for i in range(0,col_sum - col_start)]
        # 在当前sheet中逐行填补
        print('%s is interpolating' %index_name)
        temp_list = []
        col_name = sheet_temp.col_values(0)[1:419]
        for row in range(1, row_sum):
            # 获取sheet表对象某一行数据值
            row_value = sheet_temp.row_values(row)[col_start:col_sum]
            # 获取当前行中所有为空的数据位置
            print(row, row_value)
            interp_index_value = [i for i ,x in enumerate(row_value) if x != '']
            if interp_index_value == []:
                temp_list.append(row_value)
            if len(interp_index_value) <= 10:
                temp_list.append(row_value)
            else:
            # 对两侧缺失部分插值
                LR_intrep = excel_interpolation_start_end(0, col_sum - col_start, interp_index_value[0], interp_index_value[-1], row_value)
                # 对中间缺失部分进行插值
                Mid_interp = excel_interpolation_middle(x, LR_intrep, 2)
                temp_list.append(Mid_interp)
                print(Mid_interp)
        interpolation2excel(temp_list, index_name, excel_temp, col_name)
        #保存结果到本地
        print('%s has interpolated in path: F:/Learning/Yanjiusheng1/复杂网络/统计数据插补/%s.xlsx' %(index_name, index_name))
    print('All sheet have accomplished')
    return

def main(excel_path):
    # 输出excel的信息：01工作簿中sheet表数量 02工作簿中所有sheet表对象 03工作簿所有sheet表对象名称
    info_list = push_excel_information(excel_path)
    for info in info_list:
        print(info)
    # 逐sheet处理excel
    work_book = info_list[0]
    sheet_name= info_list[3]
    excel_interpolation(work_book, sheet_name)
    print('OK')
    return

if __name__ == '__main__':
    path = 'F:/Learning/Yanjiusheng1/复杂网络/统计数据插补/UsedStaData.xlsx'
    main(path)


