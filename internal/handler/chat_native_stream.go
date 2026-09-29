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

package handler

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"

	"go.uber.org/zap"

	"ragflow/internal/common"
	"ragflow/internal/service"
)

type chatCompleter interface {
	ChatCompletions(
		ctx context.Context,
		userID string,
		chatID string, sessionID string,
		messages []map[string]interface{}, question string, files []interface{},
		llmID string, genConfig map[string]interface{}, kwargs map[string]interface{},
		legacy bool,
		stream bool, streamChan chan<- service.NativeChatFrame,
	) (map[string]interface{}, error)
}

type nativeChatStreamWriter interface {
	io.Writer
	http.Flusher
}

type nativeChatFlushErrorWriter interface {
	FlushError() error
}

type nativeChatWriterUnwrapper interface {
	Unwrap() http.ResponseWriter
}

func writeNativeChatStream(ctx context.Context, writer nativeChatStreamWriter, stream <-chan service.NativeChatFrame) error {
	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case frame, ok := <-stream:
			if !ok {
				return nil
			}
			if err := ctx.Err(); err != nil {
				return err
			}
			if err := writeNativeChatFrame(writer, frame); err != nil {
				return err
			}
		}
	}
}

func writeNativeChatFrame(writer nativeChatStreamWriter, frame service.NativeChatFrame) error {
	data, err := json.Marshal(frame)
	if err != nil {
		return err
	}
	payload := []byte("data:" + addNativeJSONSpaces(data) + "\n\n")
	written, err := writer.Write(payload)
	if err != nil {
		return err
	}
	if written != len(payload) {
		return io.ErrShortWrite
	}
	if errorWriter, ok := writer.(nativeChatFlushErrorWriter); ok {
		return errorWriter.FlushError()
	}
	if unwrapper, ok := writer.(nativeChatWriterUnwrapper); ok {
		underlying := unwrapper.Unwrap()
		if errorWriter, ok := underlying.(nativeChatFlushErrorWriter); ok {
			return errorWriter.FlushError()
		}
		return http.NewResponseController(underlying).Flush()
	}
	writer.Flush()
	return nil
}

// addNativeJSONSpaces matches Python's json.dumps separators while leaving
// punctuation inside string values untouched.
func addNativeJSONSpaces(data []byte) string {
	var builder strings.Builder
	builder.Grow(len(data) + 16)
	inString := false
	escaped := false
	for _, current := range data {
		builder.WriteByte(current)
		if escaped {
			escaped = false
			continue
		}
		if inString && current == '\\' {
			escaped = true
			continue
		}
		if current == '"' {
			inString = !inString
			continue
		}
		if !inString && (current == ':' || current == ',') {
			builder.WriteByte(' ')
		}
	}
	return builder.String()
}

func logNativeChatStreamError(err error) {
	common.Warn("native chat stream write failed", zap.Error(err))
}
