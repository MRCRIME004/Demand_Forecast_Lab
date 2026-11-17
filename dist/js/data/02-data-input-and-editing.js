function parseCSV(text){
  text=text.replace(/^\uFEFF/,'');
  const first=text.split(/\r?\n/,1)[0];
  const delimiter=[';',',','\t'].sort((a,b)=>first.split(b).length-first.split(a).length)[0];
  const records=[];let row=[],field='',quoted=false;
  for(let i=0;i<text.length;i++){
    const c=text[i];
    if(c==='"'){if(quoted&&text[i+1]==='"'){field+='"';i++;}else quoted=!quoted;}
    else if(c===delimiter&&!quoted){row.push(field);field='';}
    else if((c==='\n'||c==='\r')&&!quoted){if(c==='\r'&&text[i+1]==='\n')i++;row.push(field);if(row.some(x=>x.trim()))records.push(row);row=[];field='';}
    else field+=c;
  }
  if(quoted)throw new Error('A quoted CSV value is not closed. Check the file format.');
  row.push(field);if(row.some(x=>x.trim()))records.push(row);
  if(records.length<2)throw new Error('The CSV needs a header row and at least 40 data rows.');
  const headers=records.shift().map(x=>x.trim());
  if(headers.some(x=>!x)||new Set(headers).size!==headers.length)throw new Error('Use a unique, nonblank header for every column.');
  if(records.some(r=>r.length!==headers.length))throw new Error('Some CSV rows have a different number of columns. Check separators and quoted values.');
  if(state.mode==='browser'&&records.length>BROWSER_MAX_OBSERVATIONS)throw new Error(`The hosted browser version accepts up to ${BROWSER_MAX_OBSERVATIONS.toLocaleString('en-US')} observations per run (found ${records.length.toLocaleString('en-US')} rows). Use START_WEBSITE.cmd and the local launcher for larger files, up to ${MAX_OBSERVATIONS.toLocaleString('en-US')} observations.`);
  if(records.length>MAX_OBSERVATIONS)throw new Error(`The local launcher accepts up to ${MAX_OBSERVATIONS.toLocaleString('en-US')} observations per run (found ${records.length.toLocaleString('en-US')} rows).`);
  return {headers,rows:records};
}

function loadCSV(text,name,triggerRun=false){
  const parsed=parseCSV(text);state.headers=parsed.headers;state.raw=parsed.rows;state.name=name;state.dataPage=0;
  state.fastModeUserTouched=false;
  const demand=$('demand-column'),calendar=$('calendar-column');demand.replaceChildren();calendar.replaceChildren(new Option('No calendar column: generate weekdays',''));
  state.headers.forEach((h,i)=>{demand.add(new Option(h,String(i)));calendar.add(new Option(h,String(i)));});
  const d=state.headers.findIndex(h=>['totaldemand','totalorders','targettotalorders','demand','sales','quantity'].includes(normalized(h)));
  demand.value=String(d<0?0:d);
  const c=state.headers.findIndex(h=>normalized(h).includes('dayoftheweek')||['dayofweek','weekday','day','date'].includes(normalized(h)));
  calendar.value=c<0?'':String(c);$('calendar-type').value=c>=0&&normalized(state.headers[c]).includes('date')?'date':'weekday';
  if(c>=0&&$('calendar-type').value==='date'){
    const hasWeekend=state.raw.some(row=>{const dt=new Date(String(row[c]).trim()+'T12:00:00Z');return !Number.isNaN(dt.getTime())&&[0,6].includes(dt.getUTCDay());});
    $('calendar-mode').value=hasWeekend?'all_days':'weekdays';
  }
  remap(triggerRun);error('');$('run').disabled=state.rows.length===0;
  promptDataTypeStep();
}

function remap(triggerRun=false){
  const d=Number($('demand-column').value),c=$('calendar-column').value;
  const type=$('calendar-type').value;
  state.rows=state.raw.map(row=>({demand:row[d],weekday:c!==''&&type==='weekday'?row[Number(c)]:'',date:c!==''&&type==='date'?row[Number(c)]:''}));
  $('calendar-type').disabled=c==='';$('start-weekday').disabled=c!=='';
  syncDefaultFastMode();
  updateLargeFileWarning();
  $('file-info').textContent=`${state.name} · ${state.rows.length.toLocaleString('en-US')} observations`;
  state.dataPage=0;renderInput();dirty(Boolean(triggerRun),!triggerRun);
}

for(const id of ['demand-column','calendar-column','calendar-type'])$(id).addEventListener('change',()=>remap(true));
$('calendar-mode')?.addEventListener('change',()=>{renderInput();dirty(false,true);});
for(const id of ['horizon','cv-folds','min-interval-errors','missing-treatment','max-missing-pct','outlier-treatment','outlier-threshold','hw-seasonal','hw-damped']){
  if($(id))$(id).addEventListener('change',()=>dirty(false,true));
}
if($('fast-mode')){
  $('fast-mode').addEventListener('change',()=>{
    state.fastModeUserTouched=true;
    dirty(false,true);
  });
}
if($('fast-mode-threshold')){
  $('fast-mode-threshold').addEventListener('change',()=>{
    state.fastModeUserTouched=false;
    syncDefaultFastMode();
    dirty(false,true);
  });
}
$('start-weekday').addEventListener('change',()=>{renderInput();dirty(false,true);});

async function sample(isUserAction=false){
  try{
    $('status').textContent='Loading project dataset…';
    const r=await fetch('sample.csv');
    if(!r.ok)throw new Error('The project sample could not load. Upload a CSV instead.');
    loadCSV(await r.text(),'UCI project dataset',isUserAction);
  }catch(e){error(e.message);}
}
$('sample').addEventListener('click',()=>sample(true));
$('file').addEventListener('change',async e=>{const f=e.target.files[0];if(!f)return;await receiveFile(f);e.target.value='';});

async function receiveFile(f){
  if(state.busy)return;
  error('');
  if(f.size>MAX_FILE_BYTES){
    const sizeMB=(f.size/(1024*1024)).toFixed(1);
    const limitMB=(MAX_FILE_BYTES/(1024*1024)).toFixed(0);
    const msg=`File "${f.name}" is ${sizeMB} MB, which exceeds the ${limitMB} MB maximum upload limit. Choose a CSV or TSV up to ${limitMB} MB (and up to ${MAX_OBSERVATIONS.toLocaleString('en-US')} observations).`;
    error(msg);
    $('status').textContent='Upload rejected: file exceeds the maximum size limit.';
    return;
  }
  try{
    const sizeLabel=f.size>=1_000_000?`${(f.size/(1024*1024)).toFixed(2)} MB`:`${Math.max(1,Math.round(f.size/1024))} KB`;
    $('status').textContent=`Reading ${f.name} (${sizeLabel})…`;
    $('file-info').textContent=`Loading ${f.name} (${sizeLabel})…`;
    await new Promise(resolve=>setTimeout(resolve,0));
    const text=await f.text();
    $('status').textContent=`Parsing ${f.name} (${sizeLabel})…`;
    await new Promise(resolve=>setTimeout(resolve,0));
    loadCSV(text,f.name,true);
  }catch(e){
    error(e.message);
    $('status').textContent='Could not load the uploaded file. Check the error message above.';
  }
}

$('dropzone').addEventListener('dragover',e=>{e.preventDefault();$('dropzone').classList.add('dragging');});
$('dropzone').addEventListener('dragleave',()=>$('dropzone').classList.remove('dragging'));
$('dropzone').addEventListener('drop',e=>{e.preventDefault();$('dropzone').classList.remove('dragging');if(e.dataTransfer.files[0])receiveFile(e.dataTransfer.files[0]);});

function drawPreview(){
  const holder=$('preview-plot');if(!holder)return;
  holder.replaceChildren(el('p',state.rows.length?'Your data is loaded. Open setup, confirm its type and unit, then generate the forecast report.':'Load a CSV/TSV or the project sample to begin.'));
  $('preview-heading').textContent=state.rows.length?'Dataset ready for analysis':'Start with your daily demand data';
}

function paginate(container,page,count,onChange){
  container.replaceChildren();
  if(count<=25)return;
  const total=Math.ceil(count/25);
  const prev=el('button','Previous'),next=el('button','Next');
  prev.disabled=page===0;next.disabled=page===total-1;
  prev.onclick=()=>onChange(page-1);next.onclick=()=>onChange(page+1);
  container.append(prev,el('span',`Page ${page+1} of ${total} · ${count.toLocaleString('en-US')} rows`),next);
}

function syncRowToRaw(rowIndex){
  if(!state.raw[rowIndex])return;
  const d=Number($('demand-column').value),c=$('calendar-column').value;
  const type=$('calendar-type').value;
  state.raw[rowIndex][d]=state.rows[rowIndex].demand;
  if(c!==''){
    state.raw[rowIndex][Number(c)]=type==='date'?state.rows[rowIndex].date:state.rows[rowIndex].weekday;
  }
}

function renderDataHealthBar(){
  const holder=$('data-health-bar');if(!holder)return;holder.replaceChildren();
  const x=el('div',undefined,'health-item');x.append(el('span','Loaded observations'),el('strong',`${state.rows.length.toLocaleString('en-US')} rows`));
  holder.append(x,el('p','Edit raw observations below. Recalculate to refresh the engine’s statistics and forecasts.'));
}

function renderInput(){
  const container=$('input-table');if(!container)return;
  container.replaceChildren();
  renderDataHealthBar();
  const dtype=activeDataTypeLabel();const unit=activeUnitLabel();
  const disp=displayLabelWithUnit(dtype,unit);
  const table=el('table'),thead=el('thead'),head=el('tr');
  ['Observation',disp,'Calendar','Action'].forEach(x=>head.append(el('th',x)));
  thead.append(head);table.append(thead);
  const body=el('tbody');
  const automatic=$('calendar-column').value==='';
  const isDate=$('calendar-type').value==='date'&&!automatic;

  const indexedRows=state.rows.map((r,idx)=>({r,idx})).filter(({r})=>{
    if(state.dataFilter==='blanks')return r.demand===''||r.demand===null||Number.isNaN(Number(r.demand));
    return true;
  });

  const maxPage=Math.max(0,Math.ceil(indexedRows.length/25)-1);
  if(state.dataPage>maxPage)state.dataPage=maxPage;

  indexedRows.slice(state.dataPage*25,state.dataPage*25+25).forEach(({r:row,idx:i})=>{
    const tr=el('tr');
    tr.append(el('td',String(i+1)));
    const cell=el('td'),input=el('input');
    input.type='number';input.min='0';input.step='any';input.value=row.demand;
    input.setAttribute('aria-label',`${dtype} at observation ${i+1}`);
    input.disabled=state.busy;
    input.oninput=()=>{
      row.demand=input.value;
      syncRowToRaw(i);
      if($('apply-data-edits'))$('apply-data-edits').hidden=false;
      renderDataHealthBar();
      drawPreview();
    };
    input.onchange=()=>dirty(false,true);
    cell.append(input);tr.append(cell);

    const day=el('td');
    if(automatic){
      day.textContent=weekdayNames[(Number($('start-weekday').value)-2+i)%calendarCycleLength()+2]+' (generated)';
    }else{
      const cInput=el('input');
      cInput.type=isDate?'date':'text';
      cInput.value=isDate?row.date:row.weekday;
      cInput.setAttribute('aria-label',`Calendar at observation ${i+1}`);
      cInput.disabled=state.busy;
      cInput.oninput=()=>{
        row[isDate?'date':'weekday']=cInput.value;
        syncRowToRaw(i);
        if($('apply-data-edits'))$('apply-data-edits').hidden=false;
      };
      cInput.onchange=()=>dirty(false,true);
      day.append(cInput);
    }
    tr.append(day);

    const last=el('td'),remove=el('button','Remove','remove-row');
    remove.disabled=state.busy;
    remove.setAttribute('aria-label',`Remove observation ${i+1}`);
    remove.onclick=()=>{
      state.rows.splice(i,1);
      if(state.raw.length>i)state.raw.splice(i,1);
      state.dataPage=Math.min(state.dataPage,Math.max(0,Math.ceil(state.rows.length/25)-1));
      syncDefaultFastMode();renderInput();promptDataTypeStep();
      if($('apply-data-edits'))$('apply-data-edits').hidden=false;
      dirty(false,true);
    };
    last.append(remove);tr.append(last);body.append(tr);
  });
  table.append(body);
  container.append(table);
  if(!indexedRows.length){
    container.append(el('p','No rows match this filter.','empty'));
  }
  paginate($('data-pages'),state.dataPage,indexedRows.length,p=>{state.dataPage=p;renderInput();});
  const calendarText=usesAllCalendarDays()?'Monday–Sunday calendar days':'Monday–Friday operating days';
  $('data-description').textContent=`${state.rows.length.toLocaleString('en-US')} observations of ${disp.toLowerCase()}. ${automatic?`Consecutive ${calendarText} will be generated.`:isDate?`Dates must be consecutive ${calendarText}.`:usesAllCalendarDays()?'Weekdays: Monday–Sunday names or codes 2–8.':'Weekdays: Monday–Friday names or UCI codes 2–6.'}`;
  $('file-info').textContent=`${state.name} · ${state.rows.length.toLocaleString('en-US')} observations`;
}

document.querySelectorAll('#data-row-filters button').forEach(btn=>{
  btn.addEventListener('click',()=>{
    state.dataFilter=btn.dataset.dfilter||'all';
    state.dataPage=0;
    document.querySelectorAll('#data-row-filters button').forEach(b=>b.classList.toggle('active',b===btn));
    renderInput();
  });
});

$('data-jump-input')?.addEventListener('change',()=>{
  const target=Number($('data-jump-input').value);
  if(Number.isFinite(target)&&target>=1&&target<=state.rows.length){
    state.dataFilter='all';
    document.querySelectorAll('#data-row-filters button').forEach(b=>b.classList.toggle('active',b.dataset.dfilter==='all'));
    state.dataPage=Math.floor((target-1)/25);
    renderInput();
  }
});

$('apply-data-edits')?.addEventListener('click',()=>{
  $('apply-data-edits').hidden=true;
  runAnalysis(false);
});

$('add-row').onclick=()=>{
  if(state.rows.length>=MAX_OBSERVATIONS)return error(`Maximum ${MAX_OBSERVATIONS.toLocaleString('en-US')} observations.`);
  let row={demand:state.rows.at(-1)?.demand||'0',weekday:'',date:''};
  const last=state.rows.at(-1);
  if($('calendar-column').value!==''){
    if($('calendar-type').value==='date'&&last?.date){
      const d=new Date(last.date+'T12:00:00Z');
      do{d.setUTCDate(d.getUTCDate()+1);}while(!usesAllCalendarDays()&&[0,6].includes(d.getUTCDay()));
      row.date=d.toISOString().slice(0,10);
    }else{
      const raw=String(last?.weekday||'6').toLowerCase();
      const current=Number(raw)||Object.entries(weekdayNames).find(([,v])=>v.toLowerCase().slice(0,3)===raw.slice(0,3))?.[0]||6;
      row.weekday=String(nextCalendarCode(current));
    }
  }
  state.rows.push(row);
  state.dataPage=Math.floor((state.rows.length-1)/25);
  syncDefaultFastMode();renderInput();promptDataTypeStep();
  if($('apply-data-edits'))$('apply-data-edits').hidden=false;
  dirty(false,true);
};

function csvCell(value){return '"'+String(value??'').replaceAll('"','""')+'"';}
function inputCSV(){const automatic=$('calendar-column').value==='';const dateMode=$('calendar-type').value==='date'&&!automatic;const name=dateMode?'Date':'Weekday';return [`${activeDataTypeLabel()},${name}`,...state.rows.map((r,i)=>[r.demand,automatic?weekdayNames[(Number($('start-weekday').value)-2+i)%calendarCycleLength()+2]:dateMode?r.date:r.weekday].map(csvCell).join(','))].join('\n');}
function download(blob,name){const url=URL.createObjectURL(blob),a=el('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),3000);}
function fromBase64(base64,type){const binary=atob(base64);return new Blob([Uint8Array.from(binary,c=>c.charCodeAt(0))],{type});}
$('download-input').onclick=()=>download(new Blob([inputCSV()],{type:'text/csv;charset=utf-8'}),'edited_demand.csv');

