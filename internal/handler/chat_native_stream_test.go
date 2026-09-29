package handler

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"ragflow/internal/entity"
	"ragflow/internal/service"
)

func TestWriteNativeChatStreamPreservesWireFormat(t *testing.T) {
	recorder := httptest.NewRecorder()
	stream := make(chan service.NativeChatFrame, 2)
	stream <- service.NativeChatFrame{
		Code:    0,
		Message: "",
		Data:    map[string]interface{}{"answer": "comma,colon:quoted"},
	}
	stream <- service.NativeChatFrame{Code: 0, Message: "", Data: true}
	close(stream)

	if err := writeNativeChatStream(t.Context(), recorder, stream); err != nil {
		t.Fatalf("write stream: %v", err)
	}
	want := "data:{\"code\": 0, \"message\": \"\", \"data\": {\"answer\": \"comma,colon:quoted\"}}\n\n" +
		"data:{\"code\": 0, \"message\": \"\", \"data\": true}\n\n"
	if got := recorder.Body.String(); got != want {
		t.Fatalf("wire output:\n%s\nwant:\n%s", got, want)
	}
}

func TestWriteNativeChatStreamSuccessSequence(t *testing.T) {
	recorder := httptest.NewRecorder()
	stream := make(chan service.NativeChatFrame, 3)
	stream <- service.NativeChatFrame{Code: 0, Message: "", Data: nativeChatWireAnswer{
		Answer:       "<think>reasoning",
		StartToThink: true,
		ID:           "message-1",
		SessionID:    "session-1",
	}}
	stream <- service.NativeChatFrame{Code: 0, Message: "", Data: nativeChatWireAnswer{
		Answer:    "decorated [ID:0]",
		Reference: map[string]interface{}{"chunks": []interface{}{"chunk-1"}},
		Final:     true,
		ID:        "message-1",
		SessionID: "session-1",
	}}
	stream <- service.NativeChatFrame{Code: 0, Message: "", Data: true}
	close(stream)

	if err := writeNativeChatStream(t.Context(), recorder, stream); err != nil {
		t.Fatalf("write stream: %v", err)
	}
	events := strings.Split(strings.TrimSpace(recorder.Body.String()), "\n\n")
	if len(events) != 3 {
		t.Fatalf("events=%d, want delta, final, done: %q", len(events), recorder.Body.String())
	}
	decoded := make([]map[string]interface{}, 0, len(events))
	for _, event := range events {
		var frame map[string]interface{}
		if err := json.Unmarshal([]byte(strings.TrimPrefix(event, "data:")), &frame); err != nil {
			t.Fatalf("decode event %q: %v", event, err)
		}
		decoded = append(decoded, frame)
	}
	delta := decoded[0]["data"].(map[string]interface{})
	if delta["answer"] != "<think>reasoning" || delta["start_to_think"] != true {
		t.Fatalf("delta=%#v", delta)
	}
	final := decoded[1]["data"].(map[string]interface{})
	if final["answer"] != "decorated [ID:0]" || final["final"] != true {
		t.Fatalf("final=%#v", final)
	}
	if reference, ok := final["reference"].(map[string]interface{}); !ok || len(reference["chunks"].([]interface{})) != 1 {
		t.Fatalf("final reference=%#v", final["reference"])
	}
	if decoded[2]["data"] != true {
		t.Fatalf("done=%#v", decoded[2])
	}
}

func TestChatSessionHandlerNativeStreamPrepareErrorBaseline(t *testing.T) {
	gin.SetMode(gin.TestMode)
	completer := &stubChatCompleter{run: func(ctx context.Context, stream chan<- service.NativeChatFrame) {
		select {
		case stream <- service.NativeChatFrame{
			Code:    500,
			Message: "pipeline unavailable",
			Data: map[string]interface{}{
				"answer":    "**ERROR**: pipeline unavailable",
				"reference": []interface{}{},
			},
		}:
		case <-ctx.Done():
		}
	}}
	recorder := httptest.NewRecorder()
	ctx := newNativeChatHandlerContext(recorder, context.Background(), `{"messages":[{"role":"user","content":"hi"}]}`)
	handler := &ChatSessionHandler{chatCompleter: completer}

	handler.ChatCompletions(ctx)

	if !completer.called {
		t.Fatal("chat completer was not called")
	}
	if got := recorder.Header().Get("Content-Type"); got != "text/event-stream" {
		t.Fatalf("Content-Type=%q", got)
	}
	body := recorder.Body.String()
	if !strings.Contains(body, `"code": 500`) || !strings.Contains(body, `pipeline unavailable`) {
		t.Fatalf("prepare error body=%q", body)
	}
	if strings.Contains(body, `"data": true`) {
		t.Fatalf("prepare error unexpectedly has done frame: %q", body)
	}
}

func TestChatSessionHandlerRejectsInvalidRequestBeforeSSE(t *testing.T) {
	for _, test := range []struct {
		name string
		body string
	}{
		{name: "invalid JSON", body: `{"messages":`},
		{name: "invalid shape", body: `{"messages":{}}`},
	} {
		t.Run(test.name, func(t *testing.T) {
			gin.SetMode(gin.TestMode)
			completer := &stubChatCompleter{}
			recorder := httptest.NewRecorder()
			ctx := newNativeChatHandlerContext(recorder, context.Background(), test.body)
			handler := &ChatSessionHandler{chatCompleter: completer}

			handler.ChatCompletions(ctx)

			if completer.called {
				t.Fatal("chat completer called for invalid request")
			}
			if got := recorder.Header().Get("Content-Type"); strings.HasPrefix(got, "text/event-stream") {
				t.Fatalf("invalid request entered SSE mode: Content-Type=%q", got)
			}
			if !strings.Contains(recorder.Body.String(), `"code":400`) {
				t.Fatalf("invalid request response=%q", recorder.Body.String())
			}
		})
	}
}

func TestChatSessionHandlerNativeStreamFailuresCancelProducer(t *testing.T) {
	tests := []struct {
		name      string
		writeErr  error
		flushErr  error
		wantBytes bool
	}{
		{name: "write failure", writeErr: errors.New("write failed")},
		{name: "flush failure", flushErr: errors.New("flush failed"), wantBytes: true},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			gin.SetMode(gin.TestMode)
			producerCanceled := make(chan struct{})
			completer := &stubChatCompleter{run: func(ctx context.Context, stream chan<- service.NativeChatFrame) {
				select {
				case stream <- service.NativeChatFrame{Code: 0, Message: "", Data: map[string]interface{}{"answer": "partial"}}:
				case <-ctx.Done():
					close(producerCanceled)
					return
				}
				<-ctx.Done()
				close(producerCanceled)
			}}
			writer := &nativeChatFaultWriter{header: make(http.Header), writeErr: test.writeErr, flushErr: test.flushErr}
			ctx := newNativeChatHandlerContext(writer, context.Background(), `{"messages":[{"role":"user","content":"hi"}]}`)
			handler := &ChatSessionHandler{chatCompleter: completer}

			handler.ChatCompletions(ctx)

			select {
			case <-producerCanceled:
			case <-time.After(time.Second):
				t.Fatal("producer did not observe cancellation")
			}
			if test.wantBytes != (writer.body.Len() > 0) {
				t.Fatalf("written bytes=%d, wantBytes=%v", writer.body.Len(), test.wantBytes)
			}
			if strings.Contains(writer.body.String(), `"data": true`) {
				t.Fatalf("failure path wrote done: %q", writer.body.String())
			}
		})
	}
}

func TestChatSessionHandlerNativeStreamClientCancellationStopsProducer(t *testing.T) {
	gin.SetMode(gin.TestMode)
	started := make(chan struct{})
	producerCanceled := make(chan struct{})
	completer := &stubChatCompleter{run: func(ctx context.Context, _ chan<- service.NativeChatFrame) {
		close(started)
		<-ctx.Done()
		close(producerCanceled)
	}}
	requestContext, cancelRequest := context.WithCancel(context.Background())
	recorder := httptest.NewRecorder()
	ctx := newNativeChatHandlerContext(recorder, requestContext, `{"messages":[{"role":"user","content":"hi"}]}`)
	handler := &ChatSessionHandler{chatCompleter: completer}
	handlerDone := make(chan struct{})
	go func() {
		defer close(handlerDone)
		handler.ChatCompletions(ctx)
	}()

	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("producer did not start")
	}
	cancelRequest()
	select {
	case <-producerCanceled:
	case <-time.After(time.Second):
		t.Fatal("producer did not observe client cancellation")
	}
	select {
	case <-handlerDone:
	case <-time.After(time.Second):
		t.Fatal("handler did not exit after client cancellation")
	}
	if recorder.Body.Len() != 0 {
		t.Fatalf("client cancellation wrote frames: %q", recorder.Body.String())
	}
}

type stubChatCompleter struct {
	mu     sync.Mutex
	called bool
	run    func(ctx context.Context, stream chan<- service.NativeChatFrame)
}

func (c *stubChatCompleter) ChatCompletions(
	ctx context.Context,
	_ string,
	_, _ string,
	_ []map[string]interface{},
	_ string,
	_ []interface{},
	_ string,
	_, _ map[string]interface{},
	_ bool,
	stream bool,
	streamChan chan<- service.NativeChatFrame,
) (map[string]interface{}, error) {
	c.mu.Lock()
	c.called = true
	c.mu.Unlock()
	if stream && c.run != nil {
		c.run(ctx, streamChan)
	}
	return nil, nil
}

func newNativeChatHandlerContext(writer http.ResponseWriter, requestContext context.Context, body string) *gin.Context {
	ctx, _ := gin.CreateTestContext(writer)
	request := httptest.NewRequest(http.MethodPost, "/api/v1/chat/completions", strings.NewReader(body)).WithContext(requestContext)
	request.Header.Set("Content-Type", "application/json")
	ctx.Request = request
	ctx.Set("user", &entity.User{ID: "user-1"})
	return ctx
}

type nativeChatFaultWriter struct {
	header   http.Header
	body     bytes.Buffer
	writeErr error
	flushErr error
	status   int
}

func (w *nativeChatFaultWriter) Header() http.Header {
	return w.header
}

func (w *nativeChatFaultWriter) Write(data []byte) (int, error) {
	if w.writeErr != nil {
		return 0, w.writeErr
	}
	return w.body.Write(data)
}

func (w *nativeChatFaultWriter) WriteHeader(status int) {
	w.status = status
}

func (w *nativeChatFaultWriter) Flush() {}

func (w *nativeChatFaultWriter) FlushError() error {
	return w.flushErr
}

type nativeChatWireAnswer struct {
	Answer       string                 `json:"answer"`
	Reference    map[string]interface{} `json:"reference,omitempty"`
	AudioBinary  interface{}            `json:"audio_binary"`
	Prompt       string                 `json:"prompt"`
	CreatedAt    float64                `json:"created_at"`
	Final        bool                   `json:"final"`
	ID           string                 `json:"id"`
	SessionID    string                 `json:"session_id"`
	StartToThink bool                   `json:"start_to_think,omitempty"`
}
