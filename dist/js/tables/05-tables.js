function compactColumns(data){
  if(state.tableFull||data.columns.length<=8)return data.columns;
  const preferred=['Validation_Rank','Operating_Week','Step_Range','Model','Model_Label','Role','Validation_MAE','Test_MAE','MAE','RMSE','MASE','sMAPE_pct','WAPE_pct','Bias','Tracking_Signal','Weekly_Total','Daily_Mean','Fold','Mean_MAE','p_value','Verdict','Observation','Weekday_Name','Action','Forecast_Date','Date','Forecast_Step','Forecast'];
  const chosen=preferred.filter(c=>data.columns.includes(c)).slice(0,8);
  return chosen.length>=3?chosen:data.columns.slice(0,8);
}

function populateTableSelect(){
  const select=$('table-select');
  if(!select||!state.result)return;
  const prevVal=select.value;
  select.replaceChildren();
  const filtered=state.result.tables.map((t,i)=>({t,i})).filter(({t})=>state.tableCategory==='All'||tableCategoryFor(t.key)===state.tableCategory);
  filtered.forEach(({t,i})=>select.add(new Option(t.label,String(i))));
  if(filtered.some(x=>String(x.i)===prevVal)){
    select.value=prevVal;
  }else if(filtered.length){
    const def=filtered.find(x=>x.t.key==='model_leaderboard')||filtered.find(x=>x.t.key==='model_metrics')||filtered[0];
    select.value=String(def.i);
  }
}

function renderTable(){
  if(!state.result||!state.result.tables.length)return;
  const idx=Number($('table-select').value||0);
  const data=state.result.tables[idx]||state.result.tables[0];
  const container=$('results-table');container.replaceChildren();
  const purpose=TABLE_PURPOSES[data.key]||`This table contains ${data.rows.length.toLocaleString('en-US')} row${data.rows.length===1?'':'s'} of numerical results. Click any column header to sort, or download the CSV.`;
  $('table-context').hidden=false;$('table-context').replaceChildren(el('strong','What this table tells you'),el('p',purpose));
  const columns=compactColumns(data);
  $('table-glossary').hidden=false;
  const glossHolder=$('table-glossary-content');
  if(glossHolder){
    glossHolder.replaceChildren();
    const gGrid=el('div',undefined,'glossary-grid');
    columns.forEach(c=>{
      const desc=COLUMN_DEFINITIONS[c]||`Values for ${c.replaceAll('_',' ').toLowerCase()}.`;
      const chip=el('div',undefined,'glossary-chip');
      chip.append(el('strong',c.replaceAll('_',' ')+': '),el('span',desc));
      gGrid.append(chip);
    });
    glossHolder.append(gGrid);
  }
  $('table-detail-toggle').disabled=data.columns.length<=8;
  $('table-detail-toggle').textContent=state.tableFull?'Show compact view':'Show full details';

  const searchQ=(state.tableSearch||'').trim().toLowerCase();
  let rowsToRender=data.rows.filter(r=>!searchQ||columns.some(c=>String(r[c]??'').toLowerCase().includes(searchQ)));
  if(state.tableSortCol&&columns.includes(state.tableSortCol)){
    const sc=state.tableSortCol;
    const dir=state.tableSortAsc?1:-1;
    rowsToRender=[...rowsToRender].sort((a,b)=>{
      const va=a[sc],vb=b[sc];
      if(typeof va==='number'&&typeof vb==='number')return (va-vb)*dir;
      return String(va??'').localeCompare(String(vb??''))*dir;
    });
  }

  const table=el('table'),head=el('tr');
  columns.forEach(c=>{
    const arrow=state.tableSortCol===c?(state.tableSortAsc?' ▲':' ▼'):'';
    const th=el('th',c.replaceAll('_',' ')+arrow,'sortable-th');
    th.onclick=()=>{
      if(state.tableSortCol===c)state.tableSortAsc=!state.tableSortAsc;
      else{state.tableSortCol=c;state.tableSortAsc=true;}
      renderTable();
    };
    head.append(th);
  });
  const thead=el('thead');thead.append(head);table.append(thead);
  const body=el('tbody');
  const winModel=state.result.summary?.selected_model;
  rowsToRender.slice(state.tablePage*25,state.tablePage*25+25).forEach(row=>{
    const tr=el('tr');
    if(row.Model&&row.Model===winModel)tr.className='winner-row';
    columns.forEach(c=>tr.append(el('td',fmtCell(row[c],c))));
    body.append(tr);
  });
  table.append(body);container.append(table);
  if(!rowsToRender.length)container.append(el('p','No matching rows in this table.','empty'));
  paginate($('table-pages'),state.tablePage,rowsToRender.length,p=>{state.tablePage=p;renderTable();});
}
$('table-select').onchange=()=>{state.tablePage=0;state.tableFull=false;state.tableSortCol=null;renderTable();};
$('table-detail-toggle').onclick=()=>{state.tableFull=!state.tableFull;renderTable();};
$('table-row-search')?.addEventListener('input',e=>{state.tableSearch=e.target.value;state.tablePage=0;renderTable();});
$('download-table').onclick=()=>{const t=state.result.tables[Number($('table-select').value)];download(new Blob([t.csv],{type:'text/csv;charset=utf-8'}),t.key+'.csv');};
$('download-all').onclick=()=>download(fromBase64(state.result.zip,'application/zip'),'demand_forecast_results.zip');

