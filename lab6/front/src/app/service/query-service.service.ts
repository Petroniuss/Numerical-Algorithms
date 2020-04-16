import { Injectable } from '@angular/core';
import { HttpClient, HttpParams, HttpResponse } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface QueryResponse {
  link: String;
  article: String;
  similarity: Number;
}

@Injectable({
  providedIn: 'root',
})
export class QueryService {
  serverURL = 'http://localhost:5000/';
  queryURL = this.serverURL + 'query';
  svdURL = this.serverURL + 'svd';

  constructor(private http: HttpClient) {}

  query(
    queryText: string,
    k_largest: number = 20,
    use_svd: boolean = true
  ): Observable<Array<QueryResponse>> {
    let params = new HttpParams()
      .set('query', queryText)
      .set('k', k_largest.toString())
      .set('svd', String(use_svd));

    return this.http.get<Array<QueryResponse>>(this.queryURL, {
      params: params,
    });
  }

  recalcSvd(k: number) {
    let params = new HttpParams().set('k', k.toString());

    return this.http.get<string>(this.svdURL, { params: params });
  }
}
