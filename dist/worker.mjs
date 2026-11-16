// All uploaded values stay in this worker. Only Python packages load from the CDN.
let ready;
async function initialize() {
  self.postMessage({type:'progress',message:'Loading the browser Python runtime. The first visit needs an internet connection.'});
  const {loadPyodide}=await import('https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide.mjs');
  const py=await loadPyodide({indexURL:'https://cdn.jsdelivr.net/pyodide/v314.0.7/full/'});
  self.postMessage({type:'progress',message:'Loading NumPy, pandas, SciPy, Matplotlib, scikit-learn and statsmodels. Please keep this tab open.'});
  try {
    await py.loadPackage(['numpy','pandas','scipy','matplotlib','scikit-learn','statsmodels']);
  } catch {
    try {
      await py.loadPackage(['numpy','pandas','scipy','matplotlib','scikit-learn']);
    } catch {
      await py.loadPackage(['numpy','pandas','scipy','matplotlib']);
    }
  }
  for(const name of ['engine.py','analysis_core.py','visualization.py']) {
    const r=await fetch('./'+name);
    if(!r.ok)throw new Error('The Python analysis files could not load. Refresh this page.');
    py.FS.writeFile('/home/pyodide/'+name,await r.text());
  }
  py.runPython('import sys\nsys.path.insert(0,"/home/pyodide")\nimport engine');
  py.globals.set('web_progress', message=>self.postMessage({type:'progress',message:String(message)}));
  return py;
}
self.onmessage=async ({data})=>{
  try {
    ready ||= initialize();
    const py=await ready;
    py.globals.set('input_json',JSON.stringify(data.payload));
    const text=await py.runPythonAsync('engine.analyze_json(input_json, web_progress)');
    self.postMessage({type:'result',result:JSON.parse(text),id:data.id});
    py.globals.delete('input_json');
  } catch(error) {
    ready=null;
    const lines=String(error.message||error).trim().split('\n');
    self.postMessage({type:'error',message:lines.at(-1)||'The browser engine could not start.',id:data.id});
  }
};
