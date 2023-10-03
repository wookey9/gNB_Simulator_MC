import ue

class Cell:
    def __init__(self, cellid, maxRb, maxPdu=2):
        self.ue_cnt = 0
        self.cellId = cellid
        self.max_RB = maxRb
        self.ue_list = []
        self.maxPdu = maxPdu
        self.sch_cnt = 0
        self.rb_utilized = 0
        self.tput = 0
        self.sr_period = 1000
        self.sr_cnt_slot = 5

    def attach_UE(self,serviceType):
        self.ue_list.append(ue.UE(len(self.ue_list),serviceType))

    def release_All(self):
        self.ue_list.clear()

    def set_maxpdu(self,maxpdu):
        self.maxPdu = maxpdu

    def schedule(self,slot):
        cell_sch_packetsize = 0
        cell_sch_rbsize = 0
        schpducnt = 0
        searchcnt = 0

        if len(self.ue_list) > 0:
            while schpducnt < self.maxPdu and (schpducnt + searchcnt < len(self.ue_list)):
                ue = self.ue_list[(self.sch_cnt + schpducnt + searchcnt) % len(self.ue_list)]
                if ue.traffic > 0:
                    if ue.aloc_rbcnt + cell_sch_rbsize <= self.max_RB:
                        sched_packetsize, sched_rbsize = ue.allocate()
                        cell_sch_packetsize += sched_packetsize
                        cell_sch_rbsize += sched_rbsize
                        schpducnt += 1
                    else:
                        break
                else:
                    searchcnt += 1

            sr_cnt = 0
            if slot % self.sr_period == 0:
                for ue in self.ue_list:
                    if ue.traffic == 0:
                        ue.scheduling_request()
                        sr_cnt += 1
                        if sr_cnt > self.sr_cnt_slot:
                            break

        self.sch_cnt += schpducnt

        self.rb_utilized += cell_sch_rbsize
        self.tput += cell_sch_packetsize

        return cell_sch_packetsize

    def get_stat(self):
        return self.rb_utilized, self.tput, self.sch_cnt

    def reset_stat(self):
        self.rb_utilized = 0
        self.tput = 0
        self.sch_cnt = 0

