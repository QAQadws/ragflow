let mockIsGo = false;

jest.mock('@/utils/backend-variant', () => ({
  pickByBackend: ({ go, python }: { go: unknown; python: unknown }) =>
    mockIsGo ? go : python,
}));

import {
  adaptDocumentFilter,
  adaptDocumentRunStatusFilter,
} from './document-filter-adapter';

describe('document status filter adapter', () => {
  beforeEach(() => {
    mockIsGo = false;
  });

  it('groups Go task statuses into the visible document status options', () => {
    mockIsGo = true;

    expect(
      adaptDocumentFilter({
        suffix: { pdf: 8 },
        metadata: {},
        ingestion_status: {
          UNSTART: 1,
          CREATED: 2,
          SCHEDULED: 1,
          RUNNING: 3,
          STOPPING: 1,
          STOPPED: 1,
          COMPLETED: 4,
          FAILED: 1,
        },
      }).run_status,
    ).toEqual({
      '0': 1,
      QUEUED: 3,
      '1': 4,
      '2': 1,
      '3': 4,
      '4': 1,
    });

    expect(adaptDocumentRunStatusFilter(['QUEUED', '1', '3'])).toEqual([
      'CREATED',
      'SCHEDULED',
      'RUNNING',
      'STOPPING',
      'COMPLETED',
    ]);
  });

  it('shows Go status options with zero counts when no task is running', () => {
    mockIsGo = true;

    expect(
      adaptDocumentFilter({
        suffix: { pdf: 2 },
        metadata: {},
        ingestion_status: { UNSTART: 2 },
      }),
    ).toMatchObject({
      suffix: { pdf: 2 },
      run_status: {
        '0': 2,
        QUEUED: 0,
        '1': 0,
        '2': 0,
        '3': 0,
        '4': 0,
      },
    });
  });

  it('keeps the Python filter response and selected status values', () => {
    const filter = {
      suffix: { docx: 2 },
      metadata: {},
      run_status: { '1': 1, '3': 1 },
    };

    expect(adaptDocumentFilter(filter)).toEqual(filter);
    expect(adaptDocumentRunStatusFilter(['1', '3'])).toEqual(['1', '3']);
  });
});
