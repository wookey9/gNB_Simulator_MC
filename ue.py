class UE:
    def __init__(self,uid, type):
        self.uid = uid
        self.service_type = type
        self.sched_cnt = 0
        self.traffic = 0

        if type == 0:
            self.aloc_rbcnt = 8
        else:
            self.aloc_rbcnt = 66

    def allocate(self):
        self.sched_cnt += 1
        sched_size = min(self.aloc_rbcnt, self.traffic)
        self.traffic -= sched_size

        return sched_size, self.aloc_rbcnt

    def scheduling_request(self):
        self.traffic = self.aloc_rbcnt * 10

