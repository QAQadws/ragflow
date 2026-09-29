package service

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"
	"time"

	"ragflow/internal/entity"
)

func TestNativeChatStreamTranslatorMapsNativeDeltas(t *testing.T) {
	reference := []interface{}{map[string]interface{}{
		"chunks": []interface{}{map[string]interface{}{"id": "chunk-1"}},
	}}

	tests := []struct {
		name         string
		result       AsyncChatResult
		wantAnswer   string
		wantStreamed string
		wantStart    bool
		wantEnd      bool
	}{
		{
			name:         "reasoning travels in answer",
			result:       AsyncChatResult{Reasoning: "inspect"},
			wantAnswer:   "inspect",
			wantStreamed: "inspect",
		},
		{
			name:         "end marker and answer share a frame",
			result:       AsyncChatResult{EndToThink: true, Answer: "visible"},
			wantAnswer:   "</think>visible",
			wantStreamed: "</think>visible",
			wantEnd:      true,
		},
		{
			name:         "start marker and reasoning share a frame",
			result:       AsyncChatResult{StartToThink: true, Reasoning: "inspect"},
			wantAnswer:   "<think>inspect",
			wantStreamed: "<think>inspect",
			wantStart:    true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			translator := newTestNativeChatTranslator(false, reference)
			translation := translator.Translate(test.result)
			if len(translation.frames) != 1 {
				t.Fatalf("frames=%d, want 1", len(translation.frames))
			}
			data := nativeFrameDataMap(t, translation.frames[0])
			if data["answer"] != test.wantAnswer {
				t.Fatalf("answer=%q, want %q", data["answer"], test.wantAnswer)
			}
			if translation.streamedAnswer != test.wantStreamed {
				t.Fatalf("streamed answer=%q, want %q", translation.streamedAnswer, test.wantStreamed)
			}
			if _, ok := data["reference"]; ok {
				t.Fatalf("intermediate frame unexpectedly contains reference: %#v", data)
			}
			if got, _ := data["start_to_think"].(bool); got != test.wantStart {
				t.Fatalf("start_to_think=%v, want %v", got, test.wantStart)
			}
			if got, _ := data["end_to_think"].(bool); got != test.wantEnd {
				t.Fatalf("end_to_think=%v, want %v", got, test.wantEnd)
			}
		})
	}
}

func TestNativeChatStreamTranslatorFinalReplacementPreservesTurnMaterial(t *testing.T) {
	reference := []interface{}{map[string]interface{}{"chunks": []interface{}{}}}
	translator := newTestNativeChatTranslator(false, reference)
	translator.Translate(AsyncChatResult{StartToThink: true, Reasoning: "draft"})
	translator.Translate(AsyncChatResult{EndToThink: true, Answer: "raw answer"})
	finalReference := map[string]interface{}{"chunks": []interface{}{"chunk-1"}, "total": 1}

	translation := translator.Translate(AsyncChatResult{
		Answer:    "decorated answer [ID:0]",
		Reference: finalReference,
		Prompt:    "prompt",
		CreatedAt: 456,
		Final:     true,
	})
	if translation.terminal == nil {
		t.Fatal("missing terminal material")
	}
	material := *translation.terminal
	if material.outcome != generationComplete {
		t.Fatalf("outcome=%v, want complete", material.outcome)
	}
	if material.streamedAnswer != "<think>draft</think>raw answer" {
		t.Fatalf("streamed answer=%q", material.streamedAnswer)
	}
	if material.finalAnswer != "decorated answer [ID:0]" {
		t.Fatalf("final answer=%q", material.finalAnswer)
	}
	if !reflect.DeepEqual(material.reference[len(material.reference)-1], finalReference) {
		t.Fatalf("material reference=%#v", material.reference)
	}

	data := nativeFrameDataMap(t, translation.frames[0])
	if data["answer"] != "decorated answer [ID:0]" || data["final"] != true {
		t.Fatalf("final data=%#v", data)
	}
	if data["prompt"] != "prompt" || data["created_at"] != float64(456) {
		t.Fatalf("final metadata=%#v", data)
	}
}

func TestNativeChatStreamTranslatorTerminalSequences(t *testing.T) {
	tests := []struct {
		name       string
		legacy     bool
		final      AsyncChatResult
		wantFrames int
		wantFailed bool
	}{
		{name: "native success", final: AsyncChatResult{Answer: "final", Final: true}, wantFrames: 1},
		{name: "native generation error", final: AsyncChatResult{Answer: "**ERROR**: failed", Final: true}, wantFrames: 1, wantFailed: true},
		{name: "legacy success", legacy: true, final: AsyncChatResult{Answer: "final", Final: true}, wantFrames: 1},
		{name: "legacy generation error", legacy: true, final: AsyncChatResult{Answer: "**ERROR**: failed", Final: true}, wantFrames: 2, wantFailed: true},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			translator := newTestNativeChatTranslator(test.legacy, nil)
			translator.Translate(AsyncChatResult{Answer: "partial"})
			translation := translator.Translate(test.final)
			if len(translation.frames) != test.wantFrames {
				t.Fatalf("frames=%d, want %d", len(translation.frames), test.wantFrames)
			}
			if translation.terminal == nil {
				t.Fatal("missing terminal material")
			}
			wantOutcome := generationComplete
			if test.wantFailed {
				wantOutcome = generationFailed
			}
			if translation.terminal.outcome != wantOutcome {
				t.Fatalf("outcome=%v, want %v", translation.terminal.outcome, wantOutcome)
			}

			frames := append(append([]NativeChatFrame(nil), translation.frames...), nativeChatDoneFrame())
			if done := frames[len(frames)-1]; done.Code != 0 || done.Data != true {
				t.Fatalf("done frame=%#v", done)
			}
			if test.legacy {
				lastData := nativeFrameDataMap(t, translation.frames[len(translation.frames)-1])
				if lastData["answer"] != "partial" || lastData["final"] != false {
					t.Fatalf("legacy final shape=%#v", lastData)
				}
			} else if test.wantFailed {
				data := nativeFrameDataMap(t, translation.frames[0])
				if data["answer"] != "**ERROR**: failed" || data["final"] != false {
					t.Fatalf("native error frame=%#v", data)
				}
				if translation.frames[0].Code != 0 {
					t.Fatalf("native generation error code=%d, want 0", translation.frames[0].Code)
				}
			}
		})
	}
}

func TestNativeChatStreamTranslatorFirstTerminalWins(t *testing.T) {
	translator := newTestNativeChatTranslator(false, nil)
	translator.Translate(AsyncChatResult{Answer: "partial"})
	first := translator.Translate(AsyncChatResult{Answer: "first", Final: true})
	if first.terminal == nil || len(first.frames) != 1 {
		t.Fatalf("first final=%#v", first)
	}

	for _, result := range []AsyncChatResult{
		{Answer: "late delta"},
		{Answer: "second", Final: true},
	} {
		translation := translator.Translate(result)
		if len(translation.frames) != 0 || translation.terminal != nil || translation.streamedAnswer != "" {
			t.Fatalf("translation after terminal=%#v", translation)
		}
	}
	if _, ok := translator.Finish(generationIncomplete); ok {
		t.Fatal("Finish produced terminal material twice")
	}
}

func TestNativeChatStreamTranslatorFinishReturnsPartialMaterialOnce(t *testing.T) {
	for _, outcome := range []generationOutcome{generationCanceled, generationIncomplete} {
		t.Run(strings.TrimPrefix(generationOutcomeName(outcome), "generation"), func(t *testing.T) {
			translator := newTestNativeChatTranslator(false, nil)
			translator.Translate(AsyncChatResult{Reasoning: "partial"})
			material, ok := translator.Finish(outcome)
			if !ok || material.outcome != outcome || material.streamedAnswer != "partial" {
				t.Fatalf("material=%#v ok=%v", material, ok)
			}
			if _, ok := translator.Finish(outcome); ok {
				t.Fatal("Finish produced material twice")
			}
		})
	}
}

func TestChatCompletionsNativeStreamRuntimeErrorStillEndsWithDone(t *testing.T) {
	pipeline := &fakePipeline{resultChan: makeResultChan(
		AsyncChatResult{Answer: "partial"},
		AsyncChatResult{Answer: "**ERROR**: generation failed", Final: true},
	)}
	service, store := newNativeChatTestService(pipeline)
	stream := make(chan NativeChatFrame, 8)

	_, err := service.ChatCompletions(
		t.Context(), "user-1", "chat-1", "session-1",
		[]map[string]interface{}{{"id": "message-1", "role": "user", "content": "question"}},
		"", nil, "", nil, nil, false, true, stream,
	)
	if err != nil {
		t.Fatalf("ChatCompletions failed: %v", err)
	}
	if len(stream) != 3 {
		t.Fatalf("frames=%d, want delta, error, done", len(stream))
	}
	<-stream
	errorFrame := <-stream
	if errorFrame.Code != 0 {
		t.Fatalf("generation error code=%d, want 0", errorFrame.Code)
	}
	errorData := nativeFrameDataMap(t, errorFrame)
	if errorData["answer"] != "**ERROR**: generation failed" || errorData["final"] != false {
		t.Fatalf("generation error data=%#v", errorData)
	}
	done := <-stream
	if done.Code != 0 || done.Data != true {
		t.Fatalf("done=%#v", done)
	}
	store.mu.Lock()
	defer store.mu.Unlock()
	if len(store.updateCalled) != 1 {
		t.Fatalf("updates=%d, want question only after generation failure", len(store.updateCalled))
	}
}

func TestChatCompletionsNativeStreamPrepareErrorHasNoDone(t *testing.T) {
	pipelineErr := errors.New("pipeline unavailable")
	pipeline := &fakePipeline{err: pipelineErr}
	service, _ := newNativeChatTestService(pipeline)
	stream := make(chan NativeChatFrame, 4)

	_, err := service.ChatCompletions(
		t.Context(), "user-1", "chat-1", "session-1",
		[]map[string]interface{}{{"id": "message-1", "role": "user", "content": "question"}},
		"", nil, "", nil, nil, false, true, stream,
	)
	if !errors.Is(err, pipelineErr) {
		t.Fatalf("error=%v, want %v", err, pipelineErr)
	}
	if len(stream) != 1 {
		t.Fatalf("frames=%d, want one error frame", len(stream))
	}
	frame := <-stream
	if frame.Code != 500 || frame.Message != pipelineErr.Error() {
		t.Fatalf("prepare error frame=%#v", frame)
	}
	if frame.Data == true {
		t.Fatal("prepare error unexpectedly emitted done")
	}
}

func TestChatCompletionsNativeStreamFirstFinalPersistsOnce(t *testing.T) {
	firstReference := map[string]interface{}{"chunks": []interface{}{"first"}}
	pipeline := &fakePipeline{resultChan: makeResultChan(
		AsyncChatResult{StartToThink: true, Reasoning: "reasoning"},
		AsyncChatResult{EndToThink: true, Answer: "raw"},
		AsyncChatResult{Answer: "first decorated", Reference: firstReference, Final: true},
		AsyncChatResult{Answer: "late delta"},
		AsyncChatResult{Answer: "second decorated", Reference: map[string]interface{}{"chunks": []interface{}{"second"}}, Final: true},
	)}
	service, store := newNativeChatTestService(pipeline)
	stream := make(chan NativeChatFrame, 16)

	_, err := service.ChatCompletions(
		t.Context(), "user-1", "chat-1", "session-1",
		[]map[string]interface{}{{"id": "message-1", "role": "user", "content": "question"}},
		"", nil, "", nil, nil, false, true, stream,
	)
	if err != nil {
		t.Fatalf("ChatCompletions failed: %v", err)
	}
	if len(stream) != 4 {
		t.Fatalf("frames=%d, want two deltas, first final, done", len(stream))
	}
	frames := make([]NativeChatFrame, 0, len(stream))
	for len(stream) > 0 {
		frames = append(frames, <-stream)
	}
	if frames[len(frames)-1].Data != true {
		t.Fatalf("last frame=%#v, want done", frames[len(frames)-1])
	}
	finalData := nativeFrameDataMap(t, frames[len(frames)-2])
	if finalData["answer"] != "first decorated" {
		t.Fatalf("terminal answer=%v", finalData["answer"])
	}

	store.mu.Lock()
	defer store.mu.Unlock()
	if len(store.updateCalled) != 2 {
		t.Fatalf("updates=%d, want question plus one assistant persistence", len(store.updateCalled))
	}
	stored := parseMessages(store.sessions["session-1"].Message)
	assistant := stored[len(stored)-1]
	if assistant["content"] != "<think>reasoning</think>raw" {
		t.Fatalf("stored raw streamed answer=%q", assistant["content"])
	}
	references := parseReferenceList(store.sessions["session-1"].Reference)
	if !reflect.DeepEqual(references[len(references)-1], firstReference) {
		t.Fatalf("stored reference=%#v", references)
	}
}

func TestChatCompletionsNativeStreamWithoutFinalDoesNotPersistAssistant(t *testing.T) {
	pipeline := &fakePipeline{resultChan: makeResultChan(AsyncChatResult{Answer: "partial"})}
	service, store := newNativeChatTestService(pipeline)
	stream := make(chan NativeChatFrame, 4)

	_, err := service.ChatCompletions(
		t.Context(), "user-1", "chat-1", "session-1",
		[]map[string]interface{}{{"id": "message-1", "role": "user", "content": "question"}},
		"", nil, "", nil, nil, false, true, stream,
	)
	if err != nil {
		t.Fatalf("ChatCompletions failed: %v", err)
	}
	if len(stream) != 2 {
		t.Fatalf("frames=%d, want partial and done", len(stream))
	}
	<-stream
	if done := <-stream; done.Data != true {
		t.Fatalf("done=%#v", done)
	}
	store.mu.Lock()
	defer store.mu.Unlock()
	if len(store.updateCalled) != 1 {
		t.Fatalf("updates=%d, want question only", len(store.updateCalled))
	}
}

func TestChatCompletionsNativeStreamBlockedConsumerCancelsProducer(t *testing.T) {
	pipeline := &cancelAwareNativeChatPipeline{
		started:  make(chan struct{}),
		canceled: make(chan struct{}),
	}
	service, store := newNativeChatTestService(pipeline)
	ctx, cancel := context.WithCancel(t.Context())
	stream := make(chan NativeChatFrame)
	done := make(chan struct{})
	go func() {
		defer close(done)
		_, _ = service.ChatCompletions(
			ctx, "user-1", "chat-1", "session-1",
			[]map[string]interface{}{{"id": "message-1", "role": "user", "content": "question"}},
			"", nil, "", nil, nil, false, true, stream,
		)
	}()

	select {
	case <-pipeline.started:
	case <-time.After(time.Second):
		t.Fatal("pipeline did not start")
	}
	cancel()
	select {
	case <-pipeline.canceled:
	case <-time.After(time.Second):
		t.Fatal("pipeline context was not canceled")
	}
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("service did not exit after cancellation")
	}

	store.mu.Lock()
	defer store.mu.Unlock()
	if len(store.updateCalled) != 1 {
		t.Fatalf("updates=%d, want question only", len(store.updateCalled))
	}
	select {
	case frame := <-stream:
		t.Fatalf("unexpected frame after blocked-consumer cancellation: %#v", frame)
	default:
	}
}

func newTestNativeChatTranslator(legacy bool, reference []interface{}) *nativeChatStreamTranslator {
	translator := newNativeChatStreamTranslator(legacy, "message-1", "session-1", "chat-1", reference)
	translator.nowSeconds = func() float64 { return 123 }
	return translator
}

func nativeFrameDataMap(t *testing.T, frame NativeChatFrame) map[string]interface{} {
	t.Helper()
	encoded, err := json.Marshal(frame.Data)
	if err != nil {
		t.Fatalf("marshal frame data: %v", err)
	}
	var data map[string]interface{}
	if err := json.Unmarshal(encoded, &data); err != nil {
		t.Fatalf("unmarshal frame data: %v", err)
	}
	return data
}

func generationOutcomeName(outcome generationOutcome) string {
	switch outcome {
	case generationCanceled:
		return "generationCanceled"
	case generationIncomplete:
		return "generationIncomplete"
	default:
		return "generationUnknown"
	}
}

func newNativeChatTestService(pipeline chatPipelineRunner) (*ChatSessionService, *fakeSessionStore) {
	store := newFakeSessionStore()
	userID := "user-1"
	store.sessions["session-1"] = &entity.ChatSession{
		ID:        "session-1",
		DialogID:  "chat-1",
		UserID:    &userID,
		Message:   json.RawMessage(`[{"role":"assistant","content":"Welcome!"}]`),
		Reference: json.RawMessage(`[]`),
	}
	store.dialogs["chat-1"] = &entity.Chat{
		ID:         "chat-1",
		TenantID:   userID,
		LLMID:      "chat@factory",
		LLMSetting: entity.JSONMap{},
	}
	store.dialogExists[userID+"|chat-1"] = true
	return &ChatSessionService{
		chatSessionDAO: store,
		userTenantDAO:  &fakeTenantStore{},
		pipeline:       pipeline,
	}, store
}

type cancelAwareNativeChatPipeline struct {
	started  chan struct{}
	canceled chan struct{}
}

func (p *cancelAwareNativeChatPipeline) AsyncChat(
	ctx context.Context,
	_ string,
	_ *entity.Chat,
	_ []map[string]interface{},
	_ bool,
	_ map[string]interface{},
) (<-chan AsyncChatResult, error) {
	results := make(chan AsyncChatResult)
	go func() {
		defer close(results)
		close(p.started)
		select {
		case results <- AsyncChatResult{Answer: "partial"}:
		case <-ctx.Done():
			close(p.canceled)
			return
		}
		<-ctx.Done()
		close(p.canceled)
	}()
	return results, nil
}
