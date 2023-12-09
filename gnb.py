import matplotlib.pyplot as plt

import cell
import pandas as pd
import numpy as np
import os
from collections import deque
import tensorflow.compat.v1 as tf
from PIL import Image
import threading
import random

class gNodeB:
    def __init__(self, N_Cell):
        self.ue_dist = [32,32,32,32,32,32,32,32]
        self.heavy_dist = [0,0,0,0,0,0,0,0]
        self.uplink_uenum = [0,0,0,0,0,0,0,0]
        self.maxpdu_list = [2,2,2,2,2,2,2,2]
        self.heavy_uid = []

        self.uplink_ratio_maxlen = 48 * 5 + 12 * 8
        self.uplink_ratio_track = deque(maxlen=self.uplink_ratio_maxlen)

        self.tdd_configuration = [1,1,1,1,1,1,1,1,0,0]

        self.n_cell = N_Cell
        self.cell_list = []
        self.gnb_tput = [0,0]
        for cellId in range(N_Cell):
            self.cell_list.append(cell.Cell(cellId, 66))

        self.episode_size = 1
        self.episode_iter = 0
        self.episode_cnt = 0
        self.running_slot = 30

        self.mobile_activity_down = pd.DataFrame({})
        self.mobile_activity_up = pd.DataFrame({})
        if os.path.exists('down_sms.pkl'):
            self.mobile_activity_down = pd.read_pickle('down_sms.pkl')

        if os.path.exists('up_sms.pkl'):
            self.mobile_activity_up = pd.read_pickle('up_sms.pkl')

        if os.path.exists('crnn_model'):
            self.crnn_model = tf.keras.models.load_model('crnn_model')


    def set_epi_size(self,size):
        self.episode_size = size
    def run(self, num_slot):
        if not(len(self.ue_dist) == len(self.maxpdu_list) == len(self.cell_list)):
            return -1

        for cid, cell in enumerate(self.cell_list):
            cell.set_maxpdu(self.maxpdu_list[cid])
            cell.reset_stat()
            cell.release_All()
            for u in range(self.ue_dist[cid]):
                if u < self.heavy_dist[cid]:
                    direction = 1
                    if u < self.uplink_uenum[cid]:
                        direction = 0
                    cell.attach_UE(1, direction)
                else:
                    direction = 1
                    if u < self.uplink_uenum[cid]:
                        direction = 0
                    cell.attach_UE(0,direction)

        self.gnb_tput = [0,0]

        '''threads = []
        for cell in self.cell_list:
            thread = threading.Thread(target=self.run_cell, args=(cell,num_slot,))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()
        for cell in self.cell_list:
            self.gnb_tput += cell.tput'''

        for slot in range(num_slot):
            for cell in self.cell_list:
                direction = self.tdd_configuration[slot % len(self.tdd_configuration)]
                self.gnb_tput[direction] += cell.schedule(slot, direction) / 10

        return 1

    def get_stat(self):
        cell_tput = []
        cell_rbutil = []
        cell_schedpdu = []
        for cell in self.cell_list:
            rbutil,tput,schedpdu = cell.get_stat(self.running_slot)
            cell_tput.append(tput)
            cell_rbutil.append(rbutil)
            cell_schedpdu.append(schedpdu)

        return self.gnb_tput, cell_tput, cell_rbutil, cell_schedpdu

    def apply_action(self, action):
        if len(action) == 2:
            minus_cell = abs(action[0] - 8) - 1
            plus_cell = abs(action[1] - 8) - 1

            if minus_cell >= 0:
                self.maxpdu_list[minus_cell] -= 1
            if plus_cell >= 0:
                self.maxpdu_list[plus_cell] += 1
        elif len(action) == len(self.maxpdu_list):
            for i,p in enumerate(action):
                self.maxpdu_list[i] = p

        self.episode_iter += 1

    def observe_state(self):
        self.run(self.running_slot)
        gnb_tput, cell_tput, cell_rb, cell_sched_pdu = self.get_stat()
        state = cell_rb + cell_tput + self.maxpdu_list + self.ue_dist + list(self.heavy_dist) + cell_sched_pdu

        done = 0
        if self.episode_iter % self.episode_size == 0:
            done = 1
            self.episode_cnt += 1
            #self.maxpdu_list = [2,2,2,2,2,2,2,2]
            self.update_env(self.episode_cnt)
        return state,gnb_tput , done

    def update_env(self, slot):
        if len(self.mobile_activity_down) > 0 and len(self.mobile_activity_up) > 0:
            down_row = self.mobile_activity_down.iloc[slot % len(self.mobile_activity_down)]
            up_row = self.mobile_activity_up.iloc[slot % len(self.mobile_activity_up)]
            row = down_row + up_row
            row = (row * 256 // row.sum()).astype(int)
            up_row = (up_row * 256 // (down_row + up_row).sum()).astype(int)
            self.ue_dist = row.astype(int).values.tolist()
            self.uplink_uenum = up_row.astype(int).values.tolist()


            self.uplink_ratio_track.append(sum(self.uplink_uenum) / sum(self.ue_dist))

        else:
            self.ue_dist = [32,32,32,32,32,32,32,32]

    def update_env_ue(self, uidlist):
        if len(self.mobile_activity) > 0:
            uidcnt = [0,0,0,0,0,0,0,0]
            lastUid = 0
            for i,uid in enumerate(uidlist):
                uidcnt[i] = uid - lastUid
            uidcnt[7] = 256 - lastUid
            self.ue_dist = uidcnt

        else:
            self.ue_dist = [32,32,32,32,32,32,32,32]
            #self.heavy_dist = [0,0,0,0,0,0,0,0]
        return self.ue_dist

    def update_env_heavy(self, heavy_ratio):
        self.heavy_dist = np.divide(np.multiply(self.ue_dist, heavy_ratio), 100)

    def update_env_uplink_ue(self, uplink_ratio):
        self.uplink_uenum = np.divide(np.multiply(self.ue_dist, uplink_ratio), 100)

    def update_tdd_configuration(self, interval):
        if os.path.exists('crnn_model'):
            data = self.convert_ulratio_to_2d()

            predicted_ulratio = self.crnn_model.predict(data)

            #np.savetxt('predicted_image.txt', predicted_ulratio[0,:,:,0], fmt='%f')

            listed_ulratio = []

            for i in range(0, 12):
                for j in range(0, 12):
                    ratio = predicted_ulratio[0,i,j,0]
                    listed_ulratio.append(1 - ratio)

            for i, tdd in enumerate(self.tdd_configuration):
                if i < (10 - max(10 * np.mean(listed_ulratio[-48:]), 1)):
                    self.tdd_configuration[i] = 1  # downlink
                else:
                    self.tdd_configuration[i] = 0  # uplink
            return listed_ulratio
        return []



    def convert_ulratio_to_2d(self):
        row = 12
        column = 12
        data = np.zeros((1,5,12,12,1))
        for a in range(5):
            width = 12
            height = 12
            for i in range(0, height):
                for j in range(0, width):
                    data[0,a,i,j,0] = (1 - self.uplink_ratio_track[(a * row * 4) + i * row + j])
            #np.savetxt('input_image_{}.txt'.format([a]), data[0,a,:,:,0], fmt='%f')

        return data

def avg_data(data, alpha):
    y = data
    if len(y) > 0:
        y_ema = [y[0], ]
        for y in y[1:]:
            y_ema.append(y_ema[-1] * alpha + y * (1 - alpha))
    return y_ema

if __name__ == '__main__':
    gnb = gNodeB(8)

    traffic_total = []
    traffic_ul = []

    tput_history_dl = []
    tput_history_ul = []
    tput_history_ulratio = []
    tput_history_ulratio_pre = []

    minutes = []
    tdd_minutes = []
    for i in range(len(gnb.mobile_activity_down)):
        minutes.append(i * 10)
        state, tput, _ = gnb.observe_state()
        traffic_total.append(sum(gnb.ue_dist))
        traffic_ul.append(sum(gnb.uplink_uenum))
        print(f'iter : {i}, tput : {tput[1]}/{tput[0]}')
        tput_history_dl.append(tput[1])
        tput_history_ul.append(tput[0])
        tput_history_ulratio.append(gnb.uplink_ratio_track[-1])

    plt.plot(minutes[300:], avg_data(tput_history_dl[300:], 0.9), label='downlink')
    plt.plot(minutes[300:], avg_data(tput_history_ul[300:], 0.9), label='uplink')
    plt.plot(minutes[300:], avg_data([x + y for x, y in zip(tput_history_dl, tput_history_ul)][300:], 0.9), label='total')
    plt.ylabel('Throughput(Mbps)')
    plt.xlabel('Minutes')
    plt.legend(loc='upper right')

    gnb = gNodeB(8)
    tput_history_dl  = []
    tput_history_ul  = []
    tput_history_ulratio = []
    tput_history_ulratio_pre = []

    minutes = []
    tdd_minutes = []
    tdd_interval = 48
    for i in range(len(gnb.mobile_activity_down)):
        minutes.append(i*10)
        state,tput,_ = gnb.observe_state()
        print(f'iter : {i}, tput : {tput[1]}/{tput[0]}')
        tput_history_dl.append(tput[1])
        tput_history_ul.append(tput[0])
        tput_history_ulratio.append(gnb.uplink_ratio_track[-1])

        if i % tdd_interval == 0 and len(gnb.uplink_ratio_track) == gnb.uplink_ratio_maxlen:

            ulratio_pre = gnb.update_tdd_configuration(tdd_interval)
            for j, ur in enumerate(ulratio_pre[-48:]):
                tdd_minutes.append((i + j) * 10)
                tput_history_ulratio_pre.append(ur)
            print(f'  predicted ul ratio : {np.mean(ulratio_pre)}/{gnb.uplink_ratio_track[-1]}')

    f = plt.figure()
    plt.plot(minutes[300:], avg_data(tput_history_dl[300:],0.9), label='downlink')
    plt.plot(minutes[300:], avg_data(tput_history_ul[300:],0.9), label='uplink')
    plt.plot(minutes[300:], avg_data([x + y for x,y in zip(tput_history_dl,tput_history_ul)][300:], 0.9), label='total')
    plt.ylabel('Throughput(Mbps)')
    plt.xlabel('Minutes')
    plt.legend(loc='upper right')

    f = plt.figure()
    plt.plot(minutes, avg_data(tput_history_ulratio,0.9))
    plt.plot(tdd_minutes, avg_data(tput_history_ulratio_pre, 0.9))
    plt.ylabel('Uplink traffic ratio (%)')
    plt.xlabel('Minutes')

    f = plt.figure()
    plt.plot(minutes[300:],avg_data(traffic_total[300:],0.9),label='total')
    plt.plot(minutes[300:],avg_data(traffic_ul[300:],0.9),label='uplink')
    plt.ylabel('# of active users')
    plt.xlabel('Minutes')
    plt.legend()
    plt.show()
