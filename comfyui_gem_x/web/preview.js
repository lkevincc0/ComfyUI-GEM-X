import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

function fileURL(file) {
    return api.apiURL(`/view?${new URLSearchParams(file)}`);
}

app.registerExtension({
    name: "GEM-X.Preview",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "GEMXInfer") return;
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            const result = message.gem_result?.[0];
            if (!result) return;
            if (!this.gemPreview) {
                const panel = document.createElement("div");
                panel.style.cssText = "display:flex;flex-direction:column;gap:8px;padding:8px;color:#ddd;background:#181818";
                this.addDOMWidget("gem_preview", "gem_preview", panel, {
                    serialize: false,
                    getValue: () => null,
                    setValue: () => {},
                });
                this.gemPreview = panel;
            }
            this.gemPreview.replaceChildren();
            if (result.files.preview) {
                const video = document.createElement("video");
                video.src = fileURL(result.files.preview);
                video.controls = true;
                video.loop = true;
                video.playsInline = true;
                video.style.cssText = "width:100%;max-height:360px;object-fit:contain";
                this.gemPreview.append(video);
            }
            const info = document.createElement("span");
            info.textContent = `${result.frames ?? "?"} frames · ${result.fps ?? "?"} fps`;
            this.gemPreview.append(info);
            for (const [name, file] of Object.entries(result.files)) {
                const link = document.createElement("a");
                link.href = fileURL(file);
                link.download = file.filename;
                link.textContent = `Download ${name}`;
                link.style.color = "#8cd6ff";
                this.gemPreview.append(link);
            }
            this.setSize([Math.max(this.size[0], 420), Math.max(this.size[1], 580)]);
            app.graph.setDirtyCanvas(true, true);
        };
    },
});
