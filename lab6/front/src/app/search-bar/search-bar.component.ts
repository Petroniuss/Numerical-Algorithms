import { Component, OnInit } from '@angular/core';
import { QueryService, QueryResponse } from '../service/query-service.service';
import { faCoffee } from '@fortawesome/free-solid-svg-icons';

@Component({
  selector: 'app-search-bar',
  templateUrl: './search-bar.component.html',
  styleUrls: ['./search-bar.component.scss'],
})
export class SearchBarComponent implements OnInit {
  constructor(private service: QueryService) {}

  faCoffee = faCoffee;

  loading: boolean = false;
  responses: Array<QueryResponse> = [];
  k: number = 16;
  useSvd: boolean = true;
  time: number = 0.0;

  k_value_approx = 100;

  ngOnInit(): void {}

  onSvdPressed() {
    this.loading = true;

    if (this.k_value_approx < 1 || this.k_value_approx >= 1000) {
      this.k_value_approx = 100;
    }

    this.service.recalcSvd(this.k_value_approx).subscribe(
      (data) => {
        console.log(data);
      },
      (error) => {
        this.loading = false;
        console.log(error);
      },
      () => (this.loading = false)
    );
  }

  onSearch(query: string) {
    this.loading = true;
    this.responses = [];

    if (this.k < 1 || this.k >= 200) {
      this.k = 16;
    }
    var startMs = new Date().getTime();
    this.service.query(query, this.k, this.useSvd).subscribe(
      (data) => {
        this.responses = data;
        this.time = (new Date().getTime() - startMs) / 1000;
      },
      (error) => {
        this.loading = false;
        console.log(error);
      },
      () => (this.loading = false)
    );
  }
}
