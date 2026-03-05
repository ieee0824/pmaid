package spinner

import (
	"fmt"
	"io"
	"sync"
	"time"
)

var frames = []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}

// Spinner displays an animated spinner while work is in progress.
type Spinner struct {
	w    io.Writer
	mu   sync.Mutex
	msg  string
	done chan struct{}
	wg   sync.WaitGroup
}

// New creates a new Spinner that writes to w.
func New(w io.Writer, msg string) *Spinner {
	return &Spinner{
		w:    w,
		msg:  msg,
		done: make(chan struct{}),
	}
}

// SetMessage updates the spinner message while it's running.
func (s *Spinner) SetMessage(msg string) {
	s.mu.Lock()
	s.msg = msg
	s.mu.Unlock()
}

// Start begins the spinner animation in a goroutine.
func (s *Spinner) Start() {
	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		i := 0
		ticker := time.NewTicker(80 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-s.done:
				fmt.Fprintf(s.w, "\r\033[K")
				return
			case <-ticker.C:
				s.mu.Lock()
				msg := s.msg
				s.mu.Unlock()
				fmt.Fprintf(s.w, "\r\033[K%s %s", frames[i%len(frames)], msg)
				i++
			}
		}
	}()
}

// Stop stops the spinner and waits for cleanup.
func (s *Spinner) Stop() {
	close(s.done)
	s.wg.Wait()
}
