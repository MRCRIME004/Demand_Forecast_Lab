function activeFigureView(figure){
  if(figure.method_views){
    const key=figure.activeMethod||figure.default_method||'SMA';
    if(figure.method_views[key])return figure.method_views[key];
  }
  return {
    title:figure.title,
    data:figure.data,
    method:null,
    accent_color:figure.accent_color||'#2563EB',
    interpretation:figure.interpretation||{shows:'',how_to_read:'',understood:'',next_step:''}
  };
}

function renderInterpretationBox(container,interp,accentColor){
  container.replaceChildren(); if(!interp)return;
  for(const [i,key,label] of [[1,'shows','What this graph is'],[2,'how_to_read','How to see and read it'],[3,'understood','What we understood'],[4,'next_step','What to do next']]){
    const d=el('details',undefined,'guide-part');d.open=i===3;
    d.append(el('summary',`${i}. ${label}`),el('p',interp[key]||'No additional interpretation is available.'));container.append(d);
  }
}

function buildMethodButtons(figure,onSelect){
  const bar=el('div',undefined,'method-buttons');
  bar.setAttribute('role','group');
  bar.setAttribute('aria-label',`Select forecasting method for ${figure.title}`);
  if(!figure.method_views)return bar;
  const current=figure.activeMethod||figure.default_method||'SMA';
  for(const [key,label] of METHOD_BUTTONS){
    if(!figure.method_views[key])continue;
    const btn=el('button',label,key===current?'active':'');
    btn.type='button';
    btn.dataset.method=key;
    const vColor=figure.method_views[key].accent_color||figure.accent_color||'#2563EB';
    if(key===current){
      btn.style.backgroundColor=vColor;
      btn.style.borderColor=vColor;
      const rgb=vColor.replace('#','').match(/../g).map(v=>parseInt(v,16)/255).map(v=>v<=.04045?v/12.92:((v+.055)/1.055)**2.4);const lum=rgb.reduce((n,v,i)=>n+v*[.2126,.7152,.0722][i],0);btn.style.color=(lum+.05)/.05>=1.05/(lum+.05)?'#000':'#fff';
    }
    btn.setAttribute('aria-pressed',String(key===current));
    btn.onclick=()=>{
      figure.activeMethod=key;
      onSelect(key);
    };
    bar.append(btn);
  }
  return bar;
}

function graphDownload(figure){
  const v=activeFigureView(figure);
  const baseName=figure.path.split('/').at(-1).replace(/\.png$/i,'');
  const suffix=v.method?`_${v.method}.png`:'.png';
  download(fromBase64(v.data.split(',')[1],'image/png'),baseName+suffix);
}

let dialogFigure;
function syncDialog(){
  if(!dialogFigure)return;
  const v=activeFigureView(dialogFigure);
  $('dialog-title').textContent=v.title;
  $('dialog-image').src=v.data;
  $('dialog-image').alt=chartAlt(dialogFigure,v); resetChartZoom(); $('open-full-size').href=URL.createObjectURL(fromBase64(v.data.split(',')[1],'image/png'));
  const holder=$('dialog-methods');
  if(holder){
    holder.replaceChildren();
    if(dialogFigure.method_views){
      const bar=buildMethodButtons(dialogFigure,()=>{
        syncDialog();
        if(dialogFigure.key==='02_evaluation')syncFeatured();
        renderOverviewCards();
        renderGallery();
      });
      holder.append(...bar.childNodes);
    }
  }
  if($('dialog-interpretation')){
    renderInterpretationBox($('dialog-interpretation'),v.interpretation||dialogFigure.interpretation,v.accent_color||dialogFigure.accent_color);
  }
}

function openGraph(figure){
  dialogFigure=figure;
  syncDialog();
  $('graph-dialog').showModal();
}
$('dialog-close').onclick=()=>$('graph-dialog').close();
$('dialog-download').onclick=()=>graphDownload(dialogFigure);
$('graph-dialog').addEventListener('click',e=>{if(e.target===$('graph-dialog')){const r=e.target.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)e.target.close();}});

function syncFeatured(){
  if(!state.result)return;
  const f=state.result.figures.find(x=>x.key==='02_evaluation');
  if(!f)return;
  const v=activeFigureView(f);
  const accent=v.accent_color||f.accent_color||'#2563EB';
  if($('featured-card'))$('featured-card').style.borderTopColor=accent;
  const image=el('img');image.src=v.data;image.alt=chartAlt(f,v);
  $('featured').replaceChildren(image);
  $('featured').onclick=()=>openGraph(f);
  if($('enlarge-featured'))$('enlarge-featured').onclick=()=>openGraph(f);
  $('download-featured').onclick=()=>graphDownload(f);
  if($('featured-title'))$('featured-title').textContent=v.title;
  const holder=$('featured-methods');
  if(holder){
    holder.replaceChildren();
    if(f.method_views){
      const bar=buildMethodButtons(f,()=>{
        syncFeatured();
        renderOverviewCards();
        renderGallery();
      });
      holder.append(...bar.childNodes);
    }
  }
  if($('featured-interpretation')){
    renderInterpretationBox($('featured-interpretation'),v.interpretation||f.interpretation,accent);
  }
}

function buildGraphCard(f,onMethodChange){
  const v=activeFigureView(f);
  const accent=v.accent_color||f.accent_color||'#2563EB';
  const article=el('article',undefined,'graph-tile graph-card');
  article.style.borderTop=`4px solid ${accent}`;

  const header=el('div',undefined,'graph-card-header');
  const badge=el('span',f.group,'graph-badge');
  badge.style.backgroundColor=accent+'18';
  badge.style.color=accent;
  const titleEl=el('h3',v.title);
  const leftHead=el('div',undefined,'graph-head-left');
  leftHead.append(badge,titleEl);

  const actions=el('div',undefined,'card-actions');
  const zoomBtn=el('button','Enlarge ⤢','secondary small-btn');
  zoomBtn.type='button';
  zoomBtn.onclick=()=>openGraph(f);
  const saveBtn=el('button','Download PNG','text-button');
  saveBtn.type='button';
  saveBtn.onclick=()=>graphDownload(f);
  actions.append(zoomBtn,saveBtn);
  header.append(leftHead,actions);

  const button=el('button',undefined,'graph-open');
  button.type='button';
  const img=el('img');
  img.src=v.data;img.alt=chartAlt(f,v);img.loading='lazy';
  button.setAttribute('aria-label','Enlarge '+v.title);
  button.onclick=()=>openGraph(f);
  button.append(img);

  const interpDetails=el('details',undefined,'graph-explanation');
  interpDetails.open=true;
  interpDetails.append(el('summary','What this graph means — How to see, read & understand it'));
  const interpBox=el('div',undefined,'interpretation-box');
  renderInterpretationBox(interpBox,v.interpretation||f.interpretation,accent);
  interpDetails.append(interpBox);

  article.append(header,button);
  if(f.method_views){
    const bar=buildMethodButtons(f,()=>{
      if(f.key==='02_evaluation')syncFeatured();
      onMethodChange();
    });
    article.append(bar);
  }
  article.append(interpDetails);
  return article;
}

function renderGraphPicker(container,figures,selectedSet,onChange){
  if(!container)return;
  container.replaceChildren();
  figures.forEach(f=>{
    const accent=f.accent_color||'#2563EB';
    const label=el('label',undefined,'picker-chip'+(selectedSet.has(f.key)?' active':''));
    if(selectedSet.has(f.key)){
      label.style.borderColor=accent;
      label.style.backgroundColor=accent+'12';
    }
    const cb=el('input');
    cb.type='checkbox';
    cb.checked=selectedSet.has(f.key);
    cb.onchange=()=>{
      if(cb.checked)selectedSet.add(f.key);
      else selectedSet.delete(f.key);
      onChange();
    };
    const dot=el('span',undefined,'picker-dot');
    dot.style.backgroundColor=accent;
    const text=el('span',f.title.split('(')[0].trim());
    label.append(cb,dot,text);
    container.append(label);
  });
}

function renderOverviewCards(){
  const container=$('overview-gallery');
  if(!container||!state.result)return;
  container.replaceChildren();
  const part1Figs=state.result.figures.filter(f=>f.part!==2);
  renderGraphPicker($('overview-graph-picker'),part1Figs,state.selectedOverviewKeys,()=>renderOverviewCards());
  const visible=part1Figs.filter(f=>state.selectedOverviewKeys.has(f.key));
  if(!visible.length){
    container.append(el('p','No graphs selected. Check any graph above or click "Select all graphs" to view report cards.','empty'));
    return;
  }
  visible.forEach(f=>{
    container.append(buildGraphCard(f,()=>{
      renderOverviewCards();
      renderGallery();
    }));
  });
}

if($('select-all-overview')){
  $('select-all-overview').onclick=()=>{
    if(!state.result)return;
    const part1Figs=state.result.figures.filter(f=>f.part!==2);
    const allSelected=part1Figs.every(f=>state.selectedOverviewKeys.has(f.key));
    if(allSelected)state.selectedOverviewKeys.clear();
    else part1Figs.forEach(f=>state.selectedOverviewKeys.add(f.key));
    $('select-all-overview').textContent=allSelected?'Select all graphs':'Clear selection';
    renderOverviewCards();
  };
}

function getGalleryCategoryPool(){
  if(!state.result)return [];
  return state.result.figures.filter(f=>{
    if(f.part===2&&!state.result.purchase_plan)return false;
    return state.graphGroup==='All'||f.group===state.graphGroup;
  });
}

function renderGallery(){
  const container=$('graph-gallery');
  if(!container)return;
  container.replaceChildren();
  if(!state.result)return;
  container.classList.toggle('two-col-cards',state.galleryCols===2);
  container.classList.toggle('single-col-cards',state.galleryCols!==2);

  const query=($('graph-search')?.value||'').trim().toLowerCase();
  const categoryPool=getGalleryCategoryPool();
  const pool=categoryPool.filter(f=>!query||`${f.title} ${f.group} ${f.key}`.toLowerCase().includes(query));
  const figures=pool.filter(f=>state.selectedGalleryKeys.has(f.key)&&(state.galleryAll||query||state.graphGroup!=='All'||KEY_CHARTS.has(f.key)));
  if($('show-all-charts')){$('show-all-charts').hidden=Boolean(state.guided);$('show-all-charts').textContent=state.galleryAll?'Show key charts':'Show all charts';$('show-all-charts').setAttribute('aria-pressed',String(Boolean(state.galleryAll)));}
  const searchMsg=$('graph-search-message');
  if(searchMsg){
    searchMsg.textContent=query&&!pool.length
      ?`That graph is not available. Supported choices in this category: ${categoryPool.map(f=>f.title.split('(')[0].trim()).join('; ') || 'none'}.`
      :query?`${pool.length} matching charts; ${figures.length} selected.`:`Showing ${figures.length} of ${categoryPool.length} charts in ${state.graphGroup}.`;
  }
  renderGraphPicker($('graph-checkboxes'),pool,state.selectedGalleryKeys,()=>renderGallery());
  const allSel=pool.length>0&&pool.every(f=>state.selectedGalleryKeys.has(f.key));
  if($('select-all-graphs'))$('select-all-graphs').textContent=allSel?'Clear selection':'Select all';
  if(!figures.length){
    container.append(el('p',query&&!pool.length?'No charts match this search. Clear the search or choose another category.':'No graphs selected in this category. Click "Select all" or check a graph above.','empty'));
    return;
  }
  figures.forEach(f=>{
    container.append(buildGraphCard(f,()=>{
      renderGallery();
      renderOverviewCards();
    }));
  });
}

$('graph-search')?.addEventListener('input',renderGallery);

$('gallery-cols-1')?.addEventListener('click',()=>{
  state.galleryCols=1;
  $('gallery-cols-1').classList.add('active-view-btn');
  $('gallery-cols-2')?.classList.remove('active-view-btn');
  renderGallery();
});
$('gallery-cols-2')?.addEventListener('click',()=>{
  state.galleryCols=2;
  $('gallery-cols-2').classList.add('active-view-btn');
  $('gallery-cols-1')?.classList.remove('active-view-btn');
  renderGallery();
});

if($('select-all-graphs')){
  $('select-all-graphs').onclick=()=>{
    if(!state.result)return;
    const pool=getGalleryCategoryPool();
    const allSelected=pool.every(f=>state.selectedGalleryKeys.has(f.key));
    if(allSelected)pool.forEach(f=>state.selectedGalleryKeys.delete(f.key));
    else pool.forEach(f=>state.selectedGalleryKeys.add(f.key));
    $('select-all-graphs').textContent=allSelected?'Select all':'Clear selection';
    renderGallery();
  };
}

