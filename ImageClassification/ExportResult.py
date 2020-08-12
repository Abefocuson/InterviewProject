import xlwt
import sys
import pylab as pl
import os
import shutil
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

'''
Result Exporter
Usages:
1. Save confusion matrix: csv + png
2. Export classification report: txt
3. Sort result

'''


class ResultExporter:
    y_true = None
    y_pred_prob = None
    y_pred = None
    confusion_matrix = None
    classification_report = None
    imagepath = None
    out_put_path = 'Outputs'

    def __init__(self, y_true, y_pred_prob=None, y_pred=None, imagepath=None, out_put_path = 'Outputs'):
        self.y_true = y_true
        self.y_pred_prob = y_pred_prob
        self.out_put_path = out_put_path
        if y_pred_prob is not None:
            self.y_pred = self.prob_to_class(y_pred_prob)
        else:
            self.y_pred = y_pred
        self.confusion_matrix = confusion_matrix(self.y_true, self.y_pred)
        self.classification_report = classification_report(self.y_true, np.array(self.y_pred))
        self.imagepath = imagepath
        self.create_folder_if_not_exists(out_put_path)

    def export_confusion_matrix(self, show=False, save=False, filename=None):
        filename = os.path.join(self.out_put_path, filename)
        try:
            if show or save:
                if save:
                    if filename[-4:] == '.xls':
                        np.savetxt(
                            filename,
                            self.confusion_matrix, delimiter=',')
                    else:
                        print("Must be saved as xls")
                    pl.savefig(filename[:-4] + '.png')
                if show:
                    pl.show()
                pl.matshow(self.confusion_matrix)
                pl.title('Confusion matrix of the classifier')
                pl.colorbar()
        except Exception:
            print(Exception)
        return self.confusion_matrix

    def export_classification_report(self, save=False, filename=None):
        filename = os.path.join(self.out_put_path, filename)
        if save:
            if filename[-4:] == '.txt':
                report_file = open(filename, 'w')
                report_file.write(self.classification_report)
                report_file.close()
            else:
                print("Must be saved as txt")
        return self.classification_report

    def export_false_result(self, wrong_pred_file):
        false_slice = list(map(lambda x, y: x != y, self.y_true, self.y_pred))
        false_image_path = [self.imagepath[i] for i, flag in enumerate(false_slice) if flag]
        false_y_test = [self.y_true[i] for i, flag in enumerate(false_slice) if flag]
        false_y_prob = [self.y_pred_prob[i] for i, flag in enumerate(false_slice) if flag]
        false_pred = [self.y_pred[i] for i, flag in enumerate(false_slice) if flag]

        false_y_prob = zip(*false_y_prob)
        data = [false_image_path, false_y_test, false_pred]
        head = ['File Path', 'Ground Truth', 'Predict']
        for idx, pred in enumerate(false_y_prob):
            data.append(pred)
            head.append('Prob_cls' + str(idx))
        self.y_pred_prob = self.y_pred_prob.transpose()
        workbook = xlwt.Workbook(encoding='utf-8')
        worksheet = workbook.add_sheet('sheet')
        start_row = 0
        if (head is not None):
            for id, item in enumerate(head):
                worksheet.write(0, id, str(item))
            start_row = 1
        for colume, sub_data in enumerate(data):
            for row, ele in enumerate(sub_data):
                worksheet.write(row + start_row, colume, str(ele))

        wrong_pred_file = os.path.join(self.out_put_path, wrong_pred_file)
        # if not os.path.isdir(os.path.dirname(wrong_pred_file)):
        #     os.makedirs(os.path.dirname(wrong_pred_file))
        workbook.save(wrong_pred_file)

    def export_false_pred_images(self, wrong_image_dir='Right_Wrong', move=False, print_res=False):
        wrong_image_dir = os.path.join(self.out_put_path,wrong_image_dir)
        self.create_folder_if_not_exists(wrong_image_dir)
        false_slice = list(map(lambda x, y: x != y, self.y_true, self.y_pred))
        false_image_path = [self.imagepath[i] for i, flag in enumerate(false_slice) if flag]
        false_pre_ans_label = [self.y_pred[i] for i, flag in enumerate(false_slice) if flag]
        right_label = [self.y_true[i] for i, flag in enumerate(false_slice) if flag]
        folder_name = ''
        for i, image in enumerate(false_image_path):
            folder_name = 'Right_Wrong' + str(right_label[i]) + '_' + str(false_pre_ans_label[i])
            target_path = os.path.join(wrong_image_dir,folder_name)
            self.create_folder_if_not_exists(target_path)
            target_path = os.path.join(target_path, os.path.basename(image))
            if move:
                shutil.move(image, target_path)
                if (print_res):
                    print("Moved %s to %s" % (image, target_path))
            else:
                shutil.copy(image, target_path)
                if (print_res):
                    print("Saved %s to %s" % (image, target_path))

    '''
    name:Excel Table Name
    sheet_name:Sheet name 
    data: a list of data array, like a two-dimension array
    head: the header of table

    Use this function like:
    export_xml_result('combined_res1.xls',
                               [l0_1, l1_1, right_1, predict_1, path_1],['Logist_0','Logist_1','Right_Label','Predict','File_Path'],'sheet')
    '''

    def export_all_results(self, filename):

        # Transpose
        self.y_pred_prob = self.y_pred_prob.transpose()
        data = [self.imagepath, self.y_true, self.y_pred]
        head = ['File Path', 'Ground Truth', 'Predict']

        for idx, pred in enumerate(self.y_pred_prob):
            data.append(pred)
            head.append('Prob_cls' + str(idx))
        # Transpose Back
        self.y_pred_prob = self.y_pred_prob.transpose()

        # Transpose, so that each line would be a prediction prob array of an image
        data = np.transpose(data)
        workbook = xlwt.Workbook(encoding='utf-8')
        worksheet = workbook.add_sheet('sheet')
        start_row = 0
        if (head is not None):
            for id, item in enumerate(head):
                worksheet.write(0, id, str(item))
            start_row = 1
        for row, sub_data in enumerate(data):
            for column, ele in enumerate(sub_data):
                worksheet.write(row + start_row, column, str(ele))

        filename = os.path.join(self.out_put_path, filename)
        workbook.save(filename)

    def prob_to_class(self, proba):
        if proba.shape[-1] > 1:
            return np.argmax(proba, axis=-1)
        else:
            return (proba > 0.5).astype('int32')

    def create_folder_if_not_exists(self, filename):
        if not os.path.isdir(filename):
            os.makedirs(filename)


def get_script_name():
    argv0_list = sys.argv[0].split("/");
    script_name = argv0_list[len(argv0_list) - 1];  # get script file name self
    script_name = script_name[0:-3];
    return script_name
