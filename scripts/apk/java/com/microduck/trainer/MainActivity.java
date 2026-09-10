package com.microduck.trainer;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.res.AssetManager;
import android.graphics.Color;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.view.View;
import android.view.ViewGroup;
import android.view.Window;
import android.view.WindowManager;
import android.webkit.CookieManager;
import android.webkit.WebChromeClient;
import android.webkit.WebResourceRequest;
import android.webkit.WebResourceResponse;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.FrameLayout;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;

/**
 * MicroDuck Trainer v2.0 – WebView-Shell.
 *
 * Die App ist ein Next.js-Static-Export (out/), komplett ins APK eingebettet.
 * Über shouldInterceptRequest wird ein virtueller HTTPS-Origin
 * (https://appassets.local) bedient, damit Secure-Context-Features
 * (WebAssembly/SIMD, WebGL, fetch, ES-Module-Worker) offline funktionieren.
 */
public class MainActivity extends Activity {

    private static final String HOST = "appassets.local";
    // aapt2 -A legt den Inhalt des Asset-Ordners direkt unter assets/ ab (ohne Präfix).
    private static final String ASSET_ROOT = "";

    private WebView webView;
    private FrameLayout rootLayout;
    private View customView;
    private WebChromeClient.CustomViewCallback customViewCallback;

    private static final Map<String, String> MIME = new HashMap<>();

    static {
        MIME.put("html", "text/html");
        MIME.put("htm", "text/html");
        MIME.put("js", "application/javascript");
        MIME.put("mjs", "application/javascript");
        MIME.put("css", "text/css");
        MIME.put("json", "application/json");
        MIME.put("map", "application/json");
        MIME.put("txt", "text/plain");
        MIME.put("wasm", "application/wasm");
        MIME.put("onnx", "application/octet-stream");
        MIME.put("stl", "application/octet-stream");
        MIME.put("bin", "application/octet-stream");
        MIME.put("glb", "model/gltf-binary");
        MIME.put("gltf", "model/gltf+json");
        MIME.put("png", "image/png");
        MIME.put("jpg", "image/jpeg");
        MIME.put("jpeg", "image/jpeg");
        MIME.put("webp", "image/webp");
        MIME.put("svg", "image/svg+xml");
        MIME.put("gif", "image/gif");
        MIME.put("ico", "image/x-icon");
        MIME.put("xml", "application/xml");
        MIME.put("woff", "font/woff");
        MIME.put("woff2", "font/woff2");
        MIME.put("ttf", "font/ttf");
        MIME.put("wav", "audio/wav");
        MIME.put("mp3", "audio/mpeg");
    }

    @SuppressLint("SetJavaScriptEnabled")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        Window w = getWindow();
        w.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        w.setStatusBarColor(Color.parseColor("#05060a"));
        w.setNavigationBarColor(Color.parseColor("#05060a"));

        rootLayout = new FrameLayout(this);
        webView = new WebView(this);
        rootLayout.addView(webView, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT));

        WebSettings s = webView.getSettings();
        s.setJavaScriptEnabled(true);
        s.setDomStorageEnabled(true);
        s.setDatabaseEnabled(true);
        s.setCacheMode(WebSettings.LOAD_DEFAULT);
        s.setMediaPlaybackRequiresUserGesture(false);
        s.setSupportZoom(false);
        s.setBuiltInZoomControls(false);
        s.setAllowFileAccess(false);
        s.setAllowContentAccess(false);
        s.setLoadWithOverviewMode(true);
        s.setUseWideViewPort(true);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            // SIMD/WASM-Features laufen in modernen WebView-Kerneln nativ.
            s.setSafeBrowsingEnabled(false);
        }
        CookieManager.getInstance().setAcceptCookie(true);

        webView.setBackgroundColor(Color.parseColor("#05060a"));
        webView.setOverScrollMode(View.OVER_SCROLL_NEVER);

        webView.setWebViewClient(new WebViewClient() {
            @Override
            public boolean shouldOverrideUrlLoading(WebView view, WebResourceRequest request) {
                Uri u = request.getUrl();
                // Nur der virtuelle Origin ist erlaubt – alles andere blockieren.
                return !(HOST.equals(u.getHost()));
            }

            @Override
            public WebResourceResponse shouldInterceptRequest(WebView view, WebResourceRequest request) {
                Uri u = request.getUrl();
                if (!HOST.equals(u.getHost())) {
                    return null;
                }
                String path = u.getPath();
                if (path == null || path.isEmpty() || "/".equals(path)) {
                    path = "/index.html";
                }
                String rel = path.startsWith("/") ? path.substring(1) : path;
                String assetPath = ASSET_ROOT + rel;

                AssetManager am = getAssets();
                try {
                    InputStream in = am.open(assetPath);
                    String ext = rel.contains(".")
                            ? rel.substring(rel.lastIndexOf('.') + 1).toLowerCase()
                            : "";
                    String mime = MIME.get(ext);
                    if (mime == null) mime = "application/octet-stream";
                    Map<String, String> headers = new HashMap<>();
                    headers.put("Access-Control-Allow-Origin", "https://" + HOST);
                    headers.put("Cache-Control", "no-cache");
                    WebResourceResponse resp = new WebResourceResponse(mime, null, in);
                    resp.setResponseHeaders(headers);
                    return resp;
                } catch (IOException e) {
                    // 404
                    try {
                        InputStream in = am.open(ASSET_ROOT + "404.html");
                        return new WebResourceResponse("text/html", null, in);
                    } catch (IOException ignore) {
                        return new WebResourceResponse("text/plain", null,
                                new ByteArrayInputStream("Not found".getBytes()));
                    }
                }
            }
        });

        webView.setWebChromeClient(new WebChromeClient() {
            // HTML5-Fullscreen (requestFullscreen aus der Web-App) – Pflicht,
            // damit der Vollbild-Button im Trainer wirklich vollflaechig wird.
            @Override
            public void onShowCustomView(View view, CustomViewCallback callback) {
                if (customView != null) {
                    callback.onCustomViewHidden();
                    return;
                }
                customView = view;
                customViewCallback = callback;
                webView.setVisibility(View.GONE);
                rootLayout.addView(view, new FrameLayout.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT));
                getWindow().addFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN);
                getWindow().getDecorView().setSystemUiVisibility(
                        View.SYSTEM_UI_FLAG_FULLSCREEN
                                | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                                | View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
                                | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
                                | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
                                | View.SYSTEM_UI_FLAG_LAYOUT_STABLE);
            }

            @Override
            public void onHideCustomView() {
                if (customView == null) return;
                rootLayout.removeView(customView);
                customView = null;
                if (customViewCallback != null) {
                    customViewCallback.onCustomViewHidden();
                    customViewCallback = null;
                }
                webView.setVisibility(View.VISIBLE);
                getWindow().clearFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN);
                getWindow().getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_LAYOUT_STABLE);
            }
        });

        setContentView(rootLayout);
        webView.loadUrl("https://" + HOST + "/index.html");
    }

    @Override
    public void onBackPressed() {
        if (customView != null) {
            // Fullscreen verlassen statt App schließen (entspricht onHideCustomView).
            if (customViewCallback != null) {
                customViewCallback.onCustomViewHidden();
            }
            return;
        }
        if (webView.canGoBack()) {
            webView.goBack();
        } else {
            super.onBackPressed();
        }
    }

    @Override
    protected void onDestroy() {
        if (webView != null) {
            webView.loadUrl("about:blank");
            webView.clearHistory();
            webView.removeAllViews();
            webView.destroy();
            webView = null;
        }
        super.onDestroy();
    }
}
