/* UI composition only. Forecast, inventory and table values come from result. */
const themeNames={light:'Light',dark:'Dark',report:'Report'};
const guideSections=[['shows','What this graph is'],['how_to_read','How to see and read it'],['understood','What we understood'],['next_step','What to do next']];

renderInterpretationBox=function(container,interpretation){
  container.replaceChildren();container.classList.add('segmented-guide');
  if(!interpretation)return;
  const controls=el('div',undefined,'guide-segments');controls.setAttribute('role','group');controls.setAttribute('aria-label','Reading guide sections');
  const panels=el('div',undefined,'guide-panels');let active='understood',all=false;
  const showAll=el('button','Show all four','guide-show-all');showAll.type='button';
  const select=(key)=>{active=key;all=false;paint();};
  const paint=()=>{
    controls.querySelectorAll('[data-guide-section]').forEach(b=>{const on=all||b.dataset.guideSection===active;b.setAttribute('aria-pressed',String(on));});
    panels.querySelectorAll('[data-guide-panel]').forEach(p=>p.hidden=!all&&p.dataset.guidePanel!==active);
    showAll.textContent=all?'Show selected section':'Show all four';showAll.setAttribute('aria-pressed',String(all));
  };
  guideSections.forEach(([key,label])=>{
    const b=el('button',label);b.type='button';b.dataset.guideSection=key;b.onclick=()=>select(key);controls.append(b);
    const p=el('section',undefined,'guide-panel');p.dataset.guidePanel=key;p.append(el('h4',label),el('p',interpretation[key]||'No additional interpretation is available.'));panels.append(p);
  });
  showAll.onclick=()=>{all=!all;paint();};container.setGuideAll=value=>{all=value;paint();};container.append(controls,panels,showAll);paint();
};

function applyTheme(theme){
  const choice=Object.hasOwn(themeNames,theme)?theme:'light';document.documentElement.dataset.theme=choice;$('theme-menu').value=choice;
  try{localStorage.setItem('dfl-theme',choice);}catch{}
}
const themeLabel=el('label',undefined,'theme-control');themeLabel.append(el('span','Theme'));const themeMenu=el('select');themeMenu.id='theme-menu';themeMenu.setAttribute('aria-label','Theme');
Object.entries(themeNames).forEach(([value,label])=>themeMenu.add(new Option(label,value)));themeMenu.onchange=()=>applyTheme(themeMenu.value);themeLabel.append(themeMenu);
const guidedLabel=el('label',undefined,'guided-control');const guidedSwitch=el('input');guidedSwitch.type='checkbox';guidedSwitch.id='guided-mode';guidedLabel.append(guidedSwitch,el('span','Guided mode'));
const printButton=el('button','Print report','secondary');printButton.id='print-report';printButton.type='button';printButton.onclick=()=>window.print();
const headerTools=el('div',undefined,'header-tools');headerTools.append(themeLabel,guidedLabel,printButton);document.querySelector('.topbar-right').prepend(headerTools);
try{applyTheme(localStorage.getItem('dfl-theme')||'light');}catch{applyTheme('light');}

const pipelineLabels=['Upload','Data & settings','Run','Forecast report','EOQ & schedule'];
document.querySelectorAll('.step-badge').forEach((old,i)=>{const b=el('button',pipelineLabels[i],old.className);b.id=old.id;b.type='button';b.dataset.pipelineStep=String(i);b.setAttribute('aria-label',`Pipeline: ${pipelineLabels[i]}`);old.replaceWith(b);b.onclick=()=>pipelineAction(i);});

const ribbon=el('section',undefined,'run-ribbon');ribbon.id='run-ribbon';ribbon.hidden=true;ribbon.setAttribute('aria-label','Current analysis summary');
const ribbonFacts=el('div',undefined,'ribbon-facts');ribbonFacts.id='ribbon-facts';const rerun=el('button','Re-run','secondary');rerun.id='ribbon-rerun';rerun.onclick=()=>runAnalysis(Boolean(state.result?.purchase_plan));
const exportZip=el('button','Export results ZIP','secondary');exportZip.id='ribbon-export';exportZip.onclick=()=>$('download-all').click();ribbon.append(ribbonFacts,rerun,exportZip);document.querySelector('.topbar').after(ribbon);
const guidedSelect=el('select');guidedSelect.id='guided-step-select';guidedSelect.setAttribute('aria-label','Guided methodology step');['Data and split','Exploration','Model leaderboard','Forecast and tests','EOQ and schedule'].forEach((v,i)=>guidedSelect.add(new Option(v,String(i))));guidedSelect.onchange=()=>guidedStep(Number(guidedSelect.value));ribbon.append(guidedSelect);

const drawerBackdrop=el('button');drawerBackdrop.type='button';drawerBackdrop.className='drawer-backdrop';drawerBackdrop.setAttribute('aria-label','Close setup');drawerBackdrop.onclick=()=>setSettingsCollapsed(true);document.querySelector('.workspace').before(drawerBackdrop);
const originalSettings=setSettingsCollapsed;
setSettingsCollapsed=function(collapsed){originalSettings(collapsed);document.body.dataset.setup=collapsed?'closed':'open';const drawer=$('settings-sidebar');if(collapsed){drawer.removeAttribute('role');drawer.removeAttribute('aria-modal');if(drawer.contains(document.activeElement))$('topbar-settings-btn').focus();}else{drawer.setAttribute('role','dialog');drawer.setAttribute('aria-modal','true');$('settings-toggle').focus();}};
setSettingsCollapsed(matchMedia('(max-width: 900px)').matches);
document.addEventListener('keydown',event=>{if(event.key==='Escape'&&document.body.dataset.setup==='open')setSettingsCollapsed(true);});
document.addEventListener('keydown',event=>{if(event.key!=='Tab'||document.body.dataset.setup!=='open')return;const controls=[...$('settings-sidebar').querySelectorAll('button,input,select,a[href],summary')].filter(x=>!x.disabled&&x.getClientRects().length);const first=controls[0],last=controls.at(-1);if(event.shiftKey&&document.activeElement===first){event.preventDefault();last.focus();}else if(!event.shiftKey&&document.activeElement===last){event.preventDefault();first.focus();}});
const setupClear=el('button','Clear data','secondary');setupClear.id='setup-clear-data';setupClear.onclick=()=>$('clear-data').click();$('sample').after(setupClear);
$('error').before($('status'));
const previousBusy=setBusy;
setBusy=function(busy){previousBusy(busy);for(const button of [rerun,exportZip,setupClear,$('clear-data')])button.disabled=busy||button===exportZip&&!state.result;};
const previousClear=clearResults;
clearResults=function(){previousClear();document.body.classList.remove('has-result');ribbon.hidden=true;$('table-catalog')?.replaceChildren();};

function pipelineAction(step){
  if(state.guided){guidedStep(step);return;}
  if(step===0||step===1||step===2){setSettingsCollapsed(false);const target=step===0?$('file'):step===1?$('data-type'):$('run');target.scrollIntoView({block:'center',behavior:'auto'});target.focus();}
  else view(step===3?'overview':'purchase');
}
const guidedKeys=[null,new Set(['17_stl_decomposition','18_stationarity','19_distribution','20_demand_acf_pacf','22_demand_calendar_heatmap']),null,new Set(['02_evaluation','03_future','06_errors','13_coverage','14_cv_folds','15_dm_test','23_actual_vs_forecast','24_error_direction_matrix']),null];
const originalPool=getGalleryCategoryPool;
getGalleryCategoryPool=function(){return originalPool().filter(f=>!state.guided||!state.guidedChartKeys||state.guidedChartKeys.has(f.key));};
function guidedStep(step){
  state.guidedStep=step;guidedSelect.value=String(step);state.guidedChartKeys=guidedKeys[step];state.graphGroup='All';
  const targets=['data','graphs','tables','graphs','purchase'];const target=targets[step];view(target);
  const allowed=new Set([target,'methods','howto']);if(step===3||step===4)allowed.add('tables');
  document.querySelectorAll('[role=tab]').forEach(b=>b.hidden=state.guided&&!allowed.has(b.dataset.view));
  document.querySelectorAll('[data-pipeline-step]').forEach(b=>{const active=Number(b.dataset.pipelineStep)===step;b.classList.toggle('guided-active',active);b.classList.toggle('active',active);if(active)b.setAttribute('aria-current','step');else b.removeAttribute('aria-current');});
  if(state.result){if(target==='graphs'){state.galleryAll=true;renderGallery();}if(target==='tables'){state.tableCategory='Leaderboard';populateTableSelect();renderTable();renderTableCatalog(state.result);}}
}
guidedSwitch.onchange=()=>{
  state.guided=guidedSwitch.checked;document.body.classList.toggle('guided-on',state.guided);const names=state.guided?['Data & split','Exploration','Models','Forecast & tests','EOQ & schedule']:pipelineLabels;
  ribbon.hidden=!state.result&&!state.guided;exportZip.disabled=!state.result;
  document.querySelectorAll('[data-pipeline-step]').forEach((b,i)=>{b.textContent=names[i];b.setAttribute('aria-label',`Pipeline: ${names[i]}`);});
  if(state.guided){state.beforeGuided={view:document.querySelector('[role=tab][aria-selected=true]')?.dataset.view||'overview',group:state.graphGroup,all:state.galleryAll};guidedStep(0);}
  else{state.guidedChartKeys=null;state.graphGroup=state.beforeGuided?.group||'All';state.galleryAll=state.beforeGuided?.all;document.querySelectorAll('[role=tab]').forEach(b=>b.hidden=false);document.querySelectorAll('[data-pipeline-step]').forEach(b=>{b.classList.remove('guided-active');b.removeAttribute('aria-current');});updateStepper(state.result?.purchase_plan?5:state.result?4:2);view(state.beforeGuided?.view||'overview');if(state.result)renderGallery();}
};

function renderNextDays(result){
  const table=result.tables.find(t=>t.key==='forecast_5_steps');const s=result.summary;const unit=s.unit_name;
  const dayLabel=s.parameters?.calendar_mode==='all_days'?'calendar days':'operating days';
  let strip=$('next-days-strip');if(!strip){strip=el('section',undefined,'next-days-strip');strip.id='next-days-strip';}
  strip.replaceChildren();strip.append(el('h3',`Next ${table?.rows.length||''} ${dayLabel}`));
  const days=el('div',undefined,'next-days-grid');
  for(const row of table?.rows||[]){const card=el('div',undefined,'day-forecast');card.append(el('span',row.Weekday_Name||'Operating day'),el('time',row.Forecast_Date||`Step ${row.Forecast_Step}`),el('strong',`${fmt(row[s.selected_model])} ${unit}`));const lo=row[`${s.selected_model}_lower_80`],hi=row[`${s.selected_model}_upper_80`];card.append(el('small',lo==null||hi==null?'Interval unavailable':`80% interval ${fmt(lo)}–${fmt(hi)}`));days.append(card);}
  strip.append(days);return strip;
}
function updateShortage(result){
  const pp=result.purchase_plan;if(!pp)return;const total=pp.forward_simulation_summary?.Immediate_Unfilled;const count=pp.forward_simulation_summary?.Days_With_Unfilled;
  const peaks=(pp.forward_simulation_rows||[]).filter(r=>Number(r.Backlog)>0).sort((a,b)=>Number(b.Backlog)-Number(a.Backlog));const peak=peaks[0];
  const forecast=result.tables.find(t=>t.key==='forecast')?.rows||[];
  const date=forecast[0]?.Forecast_Date===pp.plan_start_date?forecast.find(r=>r.Forecast_Step===peak?.Step)?.Forecast_Date:null;
  const receipts=(pp.order_schedule||[]).map(r=>r.Expected_Receipt_Date).filter(Boolean).sort();const firstReceipt=receipts[0];
  const firstReceived=(pp.forward_simulation_rows||[]).find(r=>Number(r.Received)>0);
  const earlyOnly=firstReceived&&(pp.forward_simulation_rows||[]).every(r=>!(Number(r.Immediate_Unfilled)>0)||Number(r.Step)<Number(firstReceived.Step));
  const prefix=Number(pp.current_stock)<Number(pp.reorder_point)?'Starting stock is below the reorder point; the plan shows':'The plan shows';
  const timing=earlyOnly?` before the first receipt on ${firstReceipt||'an unavailable calendar date'}.`:`. The first scheduled receipt is on ${firstReceipt||'an unavailable calendar date'}.`;
  const note=$('pp-shortage');if(!note)return;note.hidden=!(Number(total)>0);note.textContent=Number(total)>0?`${prefix} unfilled demand of ${fmt(total)} ${pp.unit_name} over ${fmt(count,0)} days (peak backlog ${fmt(peak?.Backlog)} on ${date||'an unavailable calendar date'})${timing}`:'';
  document.querySelectorAll('#pp-schedule-table tbody tr').forEach((tr,i)=>tr.classList.toggle('shortage-row',Number(pp.order_schedule[i]?.Backlog_At_Trigger)>0));
}
function enhanceV2(result){
  const s=result.summary;document.body.classList.add('has-result');ribbon.hidden=false;ribbonFacts.replaceChildren();
  const dayLabel=s.parameters?.calendar_mode==='all_days'?'calendar days':'operating days';
  const facts=[['Dataset',state.name||'Loaded data'],['Type',s.display_label],['Horizon',`${s.executive_summary.horizon_days} ${dayLabel}`],['Split',`${s.split.train} train / ${s.split.validation} validation / ${s.split.test} test`]];
  facts.forEach(([label,value])=>{const x=el('div',undefined,'ribbon-fact');x.append(el('span',label),el('strong',value));ribbonFacts.append(x);});
  let top=$('report-top');if(!top){top=el('div',undefined,'report-top');top.id='report-top';$('result-overview').prepend(top);}top.append($('executive-summary-card'),renderNextDays(result));
  let pair=$('evaluation-layout');if(!pair){pair=el('div',undefined,'evaluation-layout');pair.id='evaluation-layout';$('featured-card').before(pair);}pair.append($('featured-card'),$('leaderboard-section'));
  renderCompactLeaderboard(result);
  let metrics=$('primary-metrics');if(!metrics){metrics=document.querySelector('#result-overview .metrics');metrics.id='primary-metrics';}top.after(metrics);
  if($('secondary-report'))$('secondary-report').querySelector('summary').textContent='Exploration and model details';
  const head=$('executive-summary-card');head.querySelector('.eyebrow').textContent='SELECTED ON VALIDATION';head.querySelector('h2').textContent=s.selected_label;
  head.querySelector('p').textContent=`Typical final-test error: ${fmt(s.test_metrics.MAE)} ${s.unit_name}. Validation MAE: ${fmt(s.selected_validation_MAE)} ${s.unit_name}.`;
  const summary=el('div',undefined,'answer-summary');summary.append(el('span',`Next ${result.tables.find(t=>t.key==='forecast_5_steps')?.rows.length||''} ${dayLabel} total`),el('strong',`${fmt(s.executive_summary.next_5_days_total)} ${s.unit_name}`),el('small',`${fmt(s.executive_summary.next_5_days_mean)} ${s.unit_name}/day`));head.querySelector('.answer-summary')?.remove();head.append(summary);
  updateShortage(result);setSettingsCollapsed(true);if(state.guided)guidedStep(state.guidedStep||0);
  document.querySelectorAll('.segmented-guide').forEach(g=>g.setGuideAll?.(false));
}
function renderCompactLeaderboard(result){
  const s=result.summary,holder=$('leaderboard-section');holder.replaceChildren(el('span','MODEL COMPARISON','section-label'),el('h3','Validation leaderboard'),el('p',`MAE in ${s.unit_name}. Selection uses validation; test errors are held out.`));
  const wrap=el('div',undefined,'table-scroll'),table=el('table'),head=el('thead'),row=el('tr');
  ['Model','Validation MAE','Test MAE'].forEach(label=>row.append(el('th',label)));head.append(row);table.append(head);const body=el('tbody');
  for(const model of s.leaderboard||[]){const tr=el('tr');tr.classList.toggle('winner-row',model.Model===s.selected_model);tr.append(el('td',`${model.Model===s.selected_model?'★ ':''}${model.Model_Label}`),el('td',fmt(model.Validation_MAE)),el('td',fmt(model.Test_MAE)));body.append(tr);}
  table.append(body);wrap.append(table);const more=el('button','Open full leaderboard','secondary');more.onclick=()=>{view('tables');state.tableCategory='Leaderboard';populateTableSelect();renderTableCatalog(result);};holder.append(wrap,more);
}
const enhancePrevious=enhancePresentation;
enhancePresentation=function(result){enhancePrevious(result);enhanceV2(result);};

// Keep the existing global guide controls meaningful for segmented sections.
document.querySelectorAll('#graphs button').forEach(b=>{if(b.textContent==='Expand all guides'||b.textContent==='Collapse all guides'){const all=b.textContent==='Expand all guides';b.onclick=()=>document.querySelectorAll('.segmented-guide').forEach(g=>g.setGuideAll?.(all));}});

// Input cells/counts are user inputs; analytical summaries are engine output only.
renderDataHealthBar=function(){const holder=$('data-health-bar');if(!holder)return;holder.replaceChildren();const x=el('div',undefined,'health-item');x.append(el('span','Loaded observations'),el('strong',`${state.rows.length.toLocaleString()} rows`));holder.append(x,el('p','Edit raw observations below. Recalculate to refresh the engine’s statistics and forecasts.'));};
drawPreview=function(){const holder=$('preview-plot');if(!holder)return;holder.replaceChildren(el('p',state.rows.length?'Your data is loaded. Open setup, confirm its type and unit, then generate the forecast report.':'Load a CSV/TSV or the project sample to begin.'));$('preview-heading').textContent=state.rows.length?'Dataset ready for analysis':'Start with your daily demand data';};drawPreview();renderDataHealthBar();

const printOpenState=new Map();
window.addEventListener('beforeprint',()=>document.querySelectorAll('#overview details').forEach(d=>{printOpenState.set(d,d.open);d.open=true;}));
window.addEventListener('afterprint',()=>{for(const [d,open] of printOpenState)d.open=open;printOpenState.clear();});
