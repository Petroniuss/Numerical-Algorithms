import { TestBed } from '@angular/core/testing';

import { QueryService } from './query-service.service';

describe('QueryServiceService', () => {
  let service: QueryService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(QueryService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
