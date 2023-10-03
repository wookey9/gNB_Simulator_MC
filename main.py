import cell

class gNodeB:
    def __init__(self, N_Cell):
        self.n_cell = N_Cell
        self.cell_list = []
        self.gnb_tput = 0
        for cellId in range(N_Cell):
            self.cell_list.append(cell.Cell(cellId, 66))

    def run(self, ue_cnt_list, heavy_cnt_list, pdu_cnt_list, num_slot):
        if not(len(ue_cnt_list) == len(heavy_cnt_list) == len(pdu_cnt_list) == len(self.cell_list)):
            return -1

        for cid, cell in enumerate(self.cell_list):
            cell.set_maxpdu(pdu_cnt_list[cid])
            cell.reset_stat()
            cell.release_All()
            for u in range(ue_cnt_list[cid]):
                if u < heavy_cnt_list[cid]:
                    cell.attach_UE(1)
                else:
                    cell.attach_UE(0)

        self.gnb_tput = 0

        for slot in range(num_slot):
            for cell in self.cell_list:
                self.gnb_tput += cell.schedule(slot)

        return 1

    def get_stat(self):
        cell_tput = []
        cell_rbutil = []
        cell_schedpdu = []
        for cell in self.cell_list:
            rbutil,tput,schedpdu = cell.get_stat()
            cell_tput.append(tput)
            cell_rbutil.append(rbutil)
            cell_schedpdu.append(schedpdu)

        return self.gnb_tput, cell_tput, sum(cell_rbutil), sum(cell_schedpdu)


gnb = gNodeB(8)

ue_dist = [32,32,32,32,32,32,32,32]
heavy_dist = []
for d in ue_dist:
    heavy_dist.append(int(d / 10))
maxpdu_list = [2,2,2,2,2,2,2,2]
gnb.run(ue_dist,heavy_dist,maxpdu_list, 100000)
print("\nue: " + str(ue_dist))
print("pdu: " + str(maxpdu_list))
print(gnb.get_stat())

ue_dist = [144,16,16,16,16,16,16,16]
heavy_dist = []
for d in ue_dist:
    heavy_dist.append(int(d / 10))
maxpdu_list = [4,1,1,2,2,2,2,2]
gnb.run(ue_dist,heavy_dist,maxpdu_list, 100000)
print("\nue: " + str(ue_dist))
print("pdu: " + str(maxpdu_list))
print(gnb.get_stat())

ue_dist = [144,16,16,16,16,16,16,16]
heavy_dist = []
for d in ue_dist:
    heavy_dist.append(int(d / 10))
maxpdu_list = [2,2,2,2,2,2,2,2]
gnb.run(ue_dist,heavy_dist,maxpdu_list, 100000)
print("\nue: " + str(ue_dist))
print("pdu: " + str(maxpdu_list))
print(gnb.get_stat())
