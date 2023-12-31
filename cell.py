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
        self.sched_slotcnt = 0
        self.last_sched_ue = 0

    def attach_UE(self,serviceType,direction):
        self.ue_list.append(ue.UE(len(self.ue_list),serviceType,direction))

    def release_All(self):
        self.ue_list.clear()

    def set_maxpdu(self,maxpdu):
        self.maxPdu = maxpdu

    def schedule(self,slot,dir):
        cell_sch_packetsize = 0
        cell_sch_rbsize = 0
        schpducnt = 0
        searchcnt = 0
        uid = 0

        if len(self.ue_list) > 0:
            while schpducnt < self.maxPdu and (schpducnt + searchcnt < len(self.ue_list)):
                ue = self.ue_list[(self.sch_cnt + schpducnt + searchcnt) % len(self.ue_list)]
                if ue.direction == dir and ue.traffic > 0 and ue.is_sch_time(slot):
                    if ue.aloc_rbcnt + cell_sch_rbsize <= self.max_RB:
                        sched_packetsize, sched_rbsize = ue.allocate(slot)
                        cell_sch_packetsize += sched_packetsize
                        cell_sch_rbsize += sched_rbsize
                        schpducnt += 1
                    else:
                        break
                else:
                    searchcnt += 1

        '''for ue in self.ue_list:
            ue.scheduling_request(slot)'''
        if len(self.ue_list) > 0:
            self.last_sched_ue = (self.last_sched_ue + schpducnt + searchcnt) % len(self.ue_list)
        else:
            self.last_sched_ue = 0
        self.sch_cnt += schpducnt
        if schpducnt > 0:
            self.sched_slotcnt += 1

        self.rb_utilized += cell_sch_rbsize
        self.tput += cell_sch_packetsize

        return cell_sch_packetsize

    def get_stat(self, period):
        if self.sched_slotcnt > 0:
            return self.rb_utilized * 100 / (self.sched_slotcnt * self.max_RB), self.tput / 100, self.sch_cnt
        return 0, self.tput / 100, self.sch_cnt

    def reset_stat(self):
        self.last_sched_ue = 0
        self.rb_utilized = 0
        self.tput = 0
        self.sch_cnt = 0
        self.sched_slotcnt = 0

