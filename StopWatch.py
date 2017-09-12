#! /usr/bin/env python

import time


class StopWatchError(Exception):
    pass

class StartWhenNotStoppedException(StopWatchError):
    pass

class StopWhenNotRunningException(StopWatchError):
    pass

class TicksException(StopWatchError):
    pass

class getTimeWhileRunningException(StopWatchError):
    pass


class TickWhileRunningException(StopWatchError):
    pass


class StopWatch:

    def __init__(self, nTick = 0):
        self.times = []
        self.state = 0
        self.nTick = nTick
        self.cTick = 0

    def start(self):
        if self.state == 1:
            raise StartWhenNotStoppedException()
        self.state = 1
        self.times.append(time.time())

    def stop(self):
        if self.state == 0:
            raise StopWhenNotRunningException()
        self.state = 0
        self.times.append(time.time())

    def getTimes(self):
        return self.times

    def getTotalTime(self):
        n = len(self.times) 
        if n > 1:
            return self.times[-1] - self.times[0]
        elif n == 1:
            return time.time() - self.times[0]
        else:
            return 0

    def getCumulativeTime(self):
        if self.state == 1:
            raise getTimeWhileRunningException()
        n = len(self.times)
        c = 0
        for i in range(0, n, 2):
            c += self.times[i+1] - self.times[i]
        return c


    def tick(self):
        if self.state == 1:
            raise TickWhileRunningException()
        if self.nTick == 0:
            raise TicksException()
        else:
            self.cTick += 1
            progress = "%0.3f" % (self.cTick / float(self.nTick))
            elapsed = self.getCumulativeTime()
            print self.cTick, progress,
            eta = (elapsed / self.cTick) * self.nTick - elapsed
            print "%0.3f - %0.3f" % (elapsed, eta)


if __name__ == '__main__':
    chrono = StopWatch(nTick = 5)

    for i in range(5):
        chrono.start()
        time.sleep(1)
        chrono.stop()
        chrono.tick()

    print chrono.getTimes()
    print chrono.getTotalTime()
    print chrono.getCumulativeTime()
    print chrono.getTimes()





