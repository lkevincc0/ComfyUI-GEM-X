import {t} from './i18n.mjs';
import * as THREE from 'three';
import {OrbitControls} from './vendor/OrbitControls.mjs';
import {FBXLoader} from './vendor/FBXLoader.mjs';
const $=id=>document.getElementById(id);
export class MotionViewer {
    constructor(container) {
        this.container=container;this.scene=new THREE.Scene();this.scene.background=new THREE.Color('#eff1eb');
        this.camera=new THREE.PerspectiveCamera(38,1,.1,100000);
        this.renderer=new THREE.WebGLRenderer({antialias:true});this.renderer.setPixelRatio(Math.min(devicePixelRatio,2));
        this.renderer.outputColorSpace=THREE.SRGBColorSpace;container.append(this.renderer.domElement);
        this.controls=new OrbitControls(this.camera,this.renderer.domElement);this.controls.enableDamping=true;
        this.scene.add(new THREE.HemisphereLight(0xffffff,0x869078,2.4));
        const light=new THREE.DirectionalLight(0xffffff,2.2);light.position.set(200,400,300);this.scene.add(light);
        this.group=new THREE.Group();this.scene.add(this.group);
        this.grid=new THREE.GridHelper(1000,40,0xc6ccbd,0xe0e4d9);this.scene.add(this.grid);
        this.clock=new THREE.Clock();this.playing=false;this.duration=0;this.request=0;
        this.observer=new ResizeObserver(()=>{const {width,height}=container.getBoundingClientRect();this.renderer.setSize(width,height);this.camera.aspect=width/height;this.camera.updateProjectionMatrix();});this.observer.observe(container);
        this.renderer.setAnimationLoop(()=>{
            const delta=this.clock.getDelta();
            if(this.mixer&&this.playing)this.mixer.update(delta*this.speed);
            if(this.action){const time=this.action.time;$('timeline').value=time;$('time').textContent=`${time.toFixed(2)} / ${this.duration.toFixed(2)} s`;}
            this.controls.update();this.renderer.render(this.scene,this.camera);
        });
        this.speed=Number($('speed').value);
    }
    clear() {
        this.request++;this.playing=false;this.action=null;
        if(this.mixer){this.mixer.stopAllAction();this.mixer.uncacheRoot(this.model);this.mixer=null;}
        if(this.model){this.model.traverse(obj=>{obj.geometry?.dispose();for(const m of (Array.isArray(obj.material)?obj.material:[obj.material])){if(m){for(const v of Object.values(m))if(v?.isTexture)v.dispose();m.dispose();}}});this.group.remove(this.model);this.model=null;}
        $('play').disabled=true;$('timeline').disabled=true;$('timeline').value=0;$('time').textContent='0.00 / 0.00 s';$('play').textContent=t('播放');
    }
    async load(url,placement) {
        this.clear();const request=this.request;
        const buffer=await fetch(url).then(r=>{if(!r.ok)throw new Error(t('角色文件读取失败'));return r.arrayBuffer();});
        if(request!==this.request)return;
        const model=new FBXLoader().parse(buffer,location.origin+'/');
        const wrapper=new THREE.Group();wrapper.add(model);
        if(placement?.quaternion){const q=placement.quaternion;wrapper.quaternion.set(q.x,q.y,q.z,q.w);}
        if(placement?.scale){const s=placement.scale;wrapper.scale.set(s.x,s.y,s.z);}
        if(placement?.position){const p=placement.position;wrapper.position.set(p.x,p.y,p.z);}
        this.model=wrapper;this.group.add(wrapper);
        if(model.animations?.length){
            this.mixer=new THREE.AnimationMixer(model);this.action=this.mixer.clipAction(model.animations[0]);this.action.play();
            this.duration=model.animations[0].duration;this.mixer.update(0);
            $('timeline').max=this.duration;$('timeline').value=0;$('timeline').disabled=false;
            $('play').disabled=false;$('time').textContent=`0.00 / ${this.duration.toFixed(2)} s`;
        }
        this.frame();this.playing=false;$('play').textContent=t('播放');
    }
    frame(){
        if(!this.model)return;this.model.updateMatrixWorld(true);
        const box=new THREE.Box3().setFromObject(this.model),size=box.getSize(new THREE.Vector3()),center=box.getCenter(new THREE.Vector3());
        const span=Math.max(size.y,size.x,size.z,1),distance=span/(2*Math.tan(THREE.MathUtils.degToRad(this.camera.fov)/2))*1.5;
        this.controls.target.copy(center);this.camera.position.copy(center).add(new THREE.Vector3(distance*.2,distance*.12,distance));
        this.camera.near=Math.max(.01,span/1000);this.camera.far=span*100;this.camera.updateProjectionMatrix();
        this.controls.minDistance=span*.2;this.controls.maxDistance=span*15;this.controls.update();
        this.grid.position.y=box.min.y;this.grid.scale.setScalar(span/200);
    }
    toggle(){if(!this.action)return;this.playing=!this.playing;$('play').textContent=t(this.playing?'暂停':'播放');}
    seek(time){if(!this.action)return;this.action.time=time;this.mixer.update(0);}
    setSpeed(value){this.speed=value;}
}
