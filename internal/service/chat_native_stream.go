//
//  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//

package service

import (
	"context"
	"math"
	"strings"
	"time"
)

// NativeChatFrame is one transport-neutral frame in the native chat stream.
// The HTTP handler owns the SSE encoding of this value.
type NativeChatFrame struct {
	Code    int         `json:"code"`
	Message string      `json:"message"`
	Data    interface{} `json:"data"`
}

type generationOutcome uint8

const (
	generationComplete generationOutcome = iota
	generationFailed
	generationCanceled
	generationIncomplete
)

type nativeTurnMaterial struct {
	outcome        generationOutcome
	streamedAnswer string
	finalAnswer    string
	reference      []interface{}
}

type nativeChatTranslation struct {
	frames         []NativeChatFrame
	streamedAnswer string
	terminal       *nativeTurnMaterial
}

type nativeChatStreamTranslator struct {
	legacy     bool
	messageID  string
	sessionID  string
	chatID     string
	reference  []interface{}
	answer     strings.Builder
	terminal   bool
	material   nativeTurnMaterial
	nowSeconds func() float64
}

// nativeChatAnswer has the deterministic field order used by native streaming
// answer frames. Payload types stay private; only NativeChatFrame crosses the
// service boundary.
type nativeChatAnswer struct {
	Answer       string                 `json:"answer"`
	Reference    map[string]interface{} `json:"reference,omitempty"`
	AudioBinary  interface{}            `json:"audio_binary"`
	Prompt       string                 `json:"prompt"`
	CreatedAt    float64                `json:"created_at"`
	Final        bool                   `json:"final"`
	ID           string                 `json:"id"`
	SessionID    string                 `json:"session_id"`
	ChatID       string                 `json:"chat_id,omitempty"`
	StartToThink bool                   `json:"start_to_think,omitempty"`
	EndToThink   bool                   `json:"end_to_think,omitempty"`
	ThinkEvent   interface{}            `json:"think_event,omitempty"`
}

// nativeLegacyFinalAnswer preserves the alphabetical field order produced by
// encoding/json for the map used by the previous legacy final-frame path.
type nativeLegacyFinalAnswer struct {
	Answer      string                 `json:"answer"`
	AudioBinary interface{}            `json:"audio_binary"`
	CreatedAt   float64                `json:"created_at"`
	Final       bool                   `json:"final"`
	ID          string                 `json:"id"`
	Prompt      string                 `json:"prompt"`
	Reference   map[string]interface{} `json:"reference"`
	SessionID   string                 `json:"session_id"`
}

func newNativeChatStreamTranslator(legacy bool, messageID, sessionID, chatID string, reference []interface{}) *nativeChatStreamTranslator {
	return &nativeChatStreamTranslator{
		legacy:     legacy,
		messageID:  messageID,
		sessionID:  sessionID,
		chatID:     chatID,
		reference:  append([]interface{}(nil), reference...),
		nowSeconds: func() float64 { return float64(time.Now().UnixNano()) / 1e9 },
	}
}

// Translate synchronously maps one pipeline event into native transport
// frames and turn material. It owns all stream accumulation and terminal state.
func (t *nativeChatStreamTranslator) Translate(result AsyncChatResult) nativeChatTranslation {
	if t.terminal {
		return nativeChatTranslation{}
	}

	if result.Reference != nil && len(t.reference) > 0 {
		t.reference[len(t.reference)-1] = result.Reference
	}

	if result.Final {
		return t.translateFinal(result)
	}

	delta := ""
	if result.StartToThink {
		t.answer.WriteString("<think>")
		delta = "<think>"
	} else if result.EndToThink {
		t.answer.WriteString("</think>")
		delta = "</think>"
	}
	if result.Reasoning != "" {
		t.answer.WriteString(result.Reasoning)
		delta += result.Reasoning
	}
	if result.Answer != "" {
		t.answer.WriteString(result.Answer)
		delta += result.Answer
	}

	answer := delta
	reference := map[string]interface{}(nil)
	startToThink := result.StartToThink
	endToThink := result.EndToThink
	if t.legacy {
		answer = t.answer.String()
		reference = t.currentReference()
		startToThink = false
		endToThink = false
	}

	frame := t.answerFrame(nativeChatAnswer{
		Answer:       answer,
		Reference:    reference,
		AudioBinary:  nil,
		Prompt:       "",
		CreatedAt:    t.nowSeconds(),
		Final:        false,
		ID:           t.messageID,
		SessionID:    t.sessionID,
		ChatID:       t.chatID,
		StartToThink: startToThink,
		EndToThink:   endToThink,
		ThinkEvent:   result.ThinkEvent,
	})
	return nativeChatTranslation{
		frames:         []NativeChatFrame{frame},
		streamedAnswer: t.answer.String(),
	}
}

// Finish closes a stream that ended without a final pipeline event. A second
// call, or a call after Translate observed a final event, produces no material.
func (t *nativeChatStreamTranslator) Finish(outcome generationOutcome) (nativeTurnMaterial, bool) {
	if t.terminal {
		return nativeTurnMaterial{}, false
	}
	t.terminal = true
	t.material = nativeTurnMaterial{
		outcome:        outcome,
		streamedAnswer: t.answer.String(),
		reference:      append([]interface{}(nil), t.reference...),
	}
	return t.material, true
}

func (t *nativeChatStreamTranslator) translateFinal(result AsyncChatResult) nativeChatTranslation {
	t.terminal = true
	outcome := generationComplete
	if strings.Contains(result.Answer, "**ERROR**") {
		outcome = generationFailed
	}
	t.material = nativeTurnMaterial{
		outcome:        outcome,
		streamedAnswer: t.answer.String(),
		finalAnswer:    result.Answer,
		reference:      append([]interface{}(nil), t.reference...),
	}

	frames := make([]NativeChatFrame, 0, 2)
	if t.legacy {
		if outcome == generationFailed {
			frames = append(frames, t.answerFrame(nativeChatAnswer{
				Answer:      result.Answer,
				Reference:   t.currentReference(),
				AudioBinary: nil,
				Prompt:      "",
				CreatedAt:   t.nowSeconds(),
				Final:       false,
				ID:          t.messageID,
				SessionID:   t.sessionID,
				ChatID:      t.chatID,
			}))
		}
		frames = append(frames, NativeChatFrame{
			Code:    0,
			Message: "",
			Data: nativeLegacyFinalAnswer{
				Answer:      t.answer.String(),
				AudioBinary: nil,
				CreatedAt:   t.nowSeconds(),
				Final:       false,
				ID:          t.messageID,
				Prompt:      "",
				Reference:   t.currentReference(),
				SessionID:   t.sessionID,
			},
		})
	} else {
		answer := nativeChatAnswer{
			Answer:      result.Answer,
			Reference:   t.currentReference(),
			AudioBinary: nil,
			Prompt:      "",
			CreatedAt:   t.nowSeconds(),
			Final:       false,
			ID:          t.messageID,
			SessionID:   t.sessionID,
			ChatID:      t.chatID,
		}
		if outcome == generationComplete {
			if result.Reference != nil {
				answer.Reference = sanitizeNativeReference(result.Reference)
			}
			answer.AudioBinary = sanitizeJSONFloats(result.AudioBinary)
			answer.Prompt = result.Prompt
			if result.CreatedAt != 0 {
				answer.CreatedAt = finiteNativeTimestamp(result.CreatedAt)
			}
			answer.Final = true
		}
		frames = append(frames, t.answerFrame(answer))
	}

	material := t.material
	return nativeChatTranslation{
		frames:         frames,
		streamedAnswer: t.answer.String(),
		terminal:       &material,
	}
}

func (t *nativeChatStreamTranslator) answerFrame(answer nativeChatAnswer) NativeChatFrame {
	answer.Reference = sanitizeNativeReference(answer.Reference)
	answer.AudioBinary = sanitizeJSONFloats(answer.AudioBinary)
	answer.CreatedAt = finiteNativeTimestamp(answer.CreatedAt)
	return NativeChatFrame{Code: 0, Message: "", Data: answer}
}

func (t *nativeChatStreamTranslator) currentReference() map[string]interface{} {
	reference := map[string]interface{}{
		"chunks":   []interface{}{},
		"doc_aggs": []interface{}{},
	}
	if len(t.reference) == 0 {
		return reference
	}
	latest, ok := t.reference[len(t.reference)-1].(map[string]interface{})
	if !ok || latest == nil {
		return reference
	}
	reference = make(map[string]interface{}, len(latest)+1)
	for key, value := range latest {
		reference[key] = sanitizeJSONFloats(value)
	}
	if _, ok := reference["chunks"]; !ok {
		reference["chunks"] = []interface{}{}
	}
	return reference
}

func nativeChatErrorFrame(message string) NativeChatFrame {
	return NativeChatFrame{
		Code:    500,
		Message: message,
		Data: map[string]interface{}{
			"answer":    "**ERROR**: " + message,
			"reference": []interface{}{},
		},
	}
}

func nativeChatDoneFrame() NativeChatFrame {
	return NativeChatFrame{Code: 0, Message: "", Data: true}
}

func sanitizeNativeReference(reference map[string]interface{}) map[string]interface{} {
	if reference == nil {
		return nil
	}
	sanitized, _ := sanitizeJSONFloats(reference).(map[string]interface{})
	return sanitized
}

func finiteNativeTimestamp(value float64) float64 {
	if math.IsNaN(value) || math.IsInf(value, 0) {
		return 0
	}
	return value
}

func sendNativeChatFrame(ctx context.Context, streamChan chan<- NativeChatFrame, frame NativeChatFrame) bool {
	if ctx.Err() != nil {
		return false
	}
	select {
	case streamChan <- frame:
		return true
	case <-ctx.Done():
		return false
	}
}

func sendNativeChatFrames(ctx context.Context, streamChan chan<- NativeChatFrame, frames []NativeChatFrame) bool {
	for _, frame := range frames {
		if !sendNativeChatFrame(ctx, streamChan, frame) {
			return false
		}
	}
	return true
}

// drainNativeChatResults lets a producer that has already sent a terminal
// event finish without making the response wait for duplicate or late events.
func drainNativeChatResults(resultChan <-chan AsyncChatResult) {
	go func() {
		for range resultChan {
		}
	}()
}
