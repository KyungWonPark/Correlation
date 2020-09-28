package calc

import (
	"fmt"
	"math"
	"runtime"
	"sync"
	"time"
)

type ringMetaData struct {
	lock           sync.RWMutex
	ringIsEmpty    []bool
	ringBufferHead int
}

// PipeLine represents a compute pipeline
type PipeLine struct {
	numQueueSize   int
	numPusher      int
	numPoper       int
	jobQueue       chan int
	bufferMetaData ringMetaData
	pushCnt        int64
	pushCntLock    sync.RWMutex
	popCnt         int64
	popCntLock     sync.RWMutex
	signal         chan int
	debug          bool
}

// GetNP returns numPusher
func (p *PipeLine) GetNP() int {
	return p.numPusher
}

func (p *PipeLine) schedule() {
	for {
		select {
		case <-p.signal:
			p.numPoper = runtime.NumCPU()

			return
		default:
			p.popCntLock.RLock()
			popCnt := p.popCnt
			p.popCntLock.RUnlock()

			p.pushCntLock.RLock()
			pushCnt := p.pushCnt
			p.pushCntLock.RUnlock()

			// queue size must be larger than or equal to 2
			waterLevel := int64(math.Round(float64(p.numQueueSize) / 2))

			if p.numQueueSize >= 2 && p.pushCnt > int64(p.numQueueSize) {
				if pushCnt > popCnt+waterLevel {
					if (p.numPoper+2 <= runtime.NumCPU()) && (p.numPusher-2 > 0) {
						p.numPoper += 2
						p.numPusher -= 2
					}
				} else if popCnt+waterLevel > pushCnt {
					if (p.numPoper+2 <= runtime.NumCPU()) && (p.numPusher-2 > 0) {
						p.numPoper += 2
						p.numPusher -= 2
					}
				}
			}

			if p.debug {
				fmt.Printf("[Time: %s]\n", time.Now())
				fmt.Printf("Push Count: %d\n", pushCnt)
				fmt.Printf("Pop Count: %d\n", popCnt)
				fmt.Println()
				fmt.Printf("Num of Pushers: %d\n", p.numPusher)
				fmt.Printf("Num of Poppers: %d\n", p.numPoper)
				fmt.Println("- - - - - - - - - - - - - - - -")
				fmt.Println()
			}
		}

		time.Sleep(4 * time.Second)
	}
}

// Init returns a compute PipeLine
func Init(numQueueSize int, debug bool) *PipeLine {

	pl := PipeLine{
		numQueueSize: numQueueSize,
		numPusher:    4,
		numPoper:     runtime.NumCPU() - 4,
		jobQueue:     make(chan int, numQueueSize),
		bufferMetaData: ringMetaData{
			lock:        sync.RWMutex{},
			ringIsEmpty: make([]bool, numQueueSize),
		},
		pushCnt:     0,
		pushCntLock: sync.RWMutex{},
		popCnt:      0,
		popCntLock:  sync.RWMutex{},
		signal:      make(chan int),
		debug:       debug,
	}

	for i := 0; i < numQueueSize; i++ {
		pl.bufferMetaData.ringIsEmpty[i] = true
	}

	// go pl.schedule()

	if pl.debug {
		go pl.report()
	}

	return &pl
}

func (p *PipeLine) report() {
	for {
		select {
		case <-p.signal:
			return
		default:
			fmt.Printf("[Time: %s]\n", time.Now())
			fmt.Printf("Push Count: %d\n", p.pushCnt)
			fmt.Printf("Pop Count: %d\n", p.popCnt)
			fmt.Println()
			fmt.Printf("Ring Buffer Status: ")
			for i := 0; i < p.numQueueSize; i++ {
				if p.bufferMetaData.ringIsEmpty[i] {
					fmt.Printf("□")
				} else {
					fmt.Printf("■")
				}
			}
			fmt.Println()
			fmt.Printf("Num of Pushers: %d\n", p.numPusher)
			fmt.Printf("Num of Poppers: %d\n", p.numPoper)
			fmt.Println("- - - - - - - - - - - - - - - -")
			fmt.Println()
		}

		time.Sleep(4 * time.Second)
	}
}

// StopScheduler stops scheduler and shift all resources to popper
func (p *PipeLine) StopScheduler() {
	p.signal <- 0

	return
}

// Malloc claims a buffer element in ring buffer
func (p *PipeLine) Malloc() int {
	notFound := true
	var bufferIdx int

	p.bufferMetaData.lock.Lock()
	for i := p.bufferMetaData.ringBufferHead; notFound; i++ {
		idx := i % p.numQueueSize
		if p.bufferMetaData.ringIsEmpty[idx] {
			p.bufferMetaData.ringIsEmpty[idx] = false
			bufferIdx = idx
			p.bufferMetaData.ringBufferHead = (idx + 1) % p.numQueueSize
			notFound = false
		}
	}
	p.bufferMetaData.lock.Unlock()

	return bufferIdx
}

// Push pushes data to process into the job queue
func (p *PipeLine) Push(jobID int) {

	p.jobQueue <- jobID

	p.pushCntLock.Lock()
	p.pushCnt++
	p.pushCntLock.Unlock()

	return
}

// Close closed pipeline
func (p *PipeLine) Close() {
	close(p.jobQueue)

	return
}

// Pop pops the data from the job queue
func (p *PipeLine) Pop() (int, bool) {
	jobID, ok := <-p.jobQueue

	if ok {
		p.popCntLock.Lock()
		p.popCnt++
		p.popCntLock.Unlock()
	}

	return jobID, ok
}

// Free frees slot from ring buffer
func (p *PipeLine) Free(i int) {
	p.bufferMetaData.ringIsEmpty[i] = true

	return
}

/*
	Workflow:

	Malloc -> Push -> Pop -> Free
*/

type statistic struct {
	avg float64
	std float64
}
