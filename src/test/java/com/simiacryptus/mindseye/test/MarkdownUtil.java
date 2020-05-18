/*
 * Copyright (c) 2020 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.test;

import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import com.simiacryptus.ref.lang.RefUtil;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.filefilter.IOFileFilter;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClientBuilder;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * The type Markdown util.
 */
public class MarkdownUtil {
  /**
   * The constant logger.
   */
  protected static final Logger logger = LoggerFactory.getLogger(MarkdownUtil.class);

  /**
   * Fix hugo markdown.
   */
  @Test
  public void fixHugoMarkdown() {
    Collection<File> mdFiles = FileUtils.listFiles(new File("H:\\SimiaCryptus\\scalaJS\\blog.simiacryptus.com\\content\\posts"),
        new IOFileFilter() {
          @Override
          public boolean accept(File file) {
            return file.getName().endsWith(".md");
          }

          @Override
          public boolean accept(File dir, String name) {
            return name.endsWith(".md");
          }
        }, new IOFileFilter() {
          @Override
          public boolean accept(File file) {
            return true;
          }

          @Override
          public boolean accept(File dir, String name) {
            return true;
          }
        });
    File imageDir = new File("H:\\SimiaCryptus\\scalaJS\\blog.simiacryptus.com\\static\\img");
    mdFiles.forEach(mdFile -> {
      try {
        String data = FileUtils.readFileToString(mdFile, "UTF-8");
        data = replaceAll(data, downloadAll(imageDir, data), mdFile.toPath().getParent());
        JsonObject headerJson = getHeaderJson(data);
        boolean writeHeader = false;
        if (headerJson.has("thumbnail")) {
          logger.info(mdFile.getName() + " has a thumbnail");
        } else {
          HashMap<String, String> imageElements = allImageElements(data);
          if (!imageElements.isEmpty()) {
            Map.Entry<String, String> next = imageElements.entrySet().iterator().next();
            headerJson.addProperty("thumbnail", next.getValue());
            RefUtil.freeRef(next);
            writeHeader = true;
            logger.info(String.format("Setting thumbnail for %s", mdFile.getName()));
          }
        }
        if (!headerJson.has("comments")) {
          headerJson.addProperty("comments", true);
          writeHeader = true;
          logger.info(String.format("Enable comments for %s", mdFile.getName()));
        }
        if (writeHeader) data = setHeaderJson(data, headerJson);
        FileUtils.writeStringToFile(mdFile, data, "UTF-8");
      } catch (Throwable e) {
        e.printStackTrace();
      }
    });
  }

  /**
   * Replace all string.
   *
   * @param data      the data
   * @param downloads the downloads
   * @param base      the base
   * @return the string
   */
  @Nullable
  public String replaceAll(String data, HashMap<String, File> downloads, Path base) {
    for (Map.Entry<String, File> e : downloads.entrySet()) {
      try {
        String replacement = base.relativize(e.getValue().toPath()).toString();
        data = data.replaceAll(Pattern.quote(e.getKey()), replacement.replaceAll("\\\\", "/"));
      } catch (Throwable ex) {
        ex.printStackTrace();
      }
    }
    return data;
  }

  /**
   * Download all hash map.
   *
   * @param imageDir the image dir
   * @param data     the data
   * @return the hash map
   */
  @NotNull
  public HashMap<String, File> downloadAll(File imageDir, String data) {
    Pattern compile = Pattern.compile("\\!\\[(.*)\\]\\((.*)\\)");
    Matcher matcher = compile.matcher(data);
    HashMap<String, File> downloads = new HashMap<>();
    while (matcher.find()) {
      String url = matcher.group(2);
      try {
        File localFile = download(url, imageDir);
        if (null != localFile) downloads.put(url, localFile);
      } catch (Throwable e) {
        e.printStackTrace();
      }
    }
    return downloads;
  }

  /**
   * All image elements hash map.
   *
   * @param data the data
   * @return the hash map
   */
  @NotNull
  public HashMap<String, String> allImageElements(String data) {
    Pattern compile = Pattern.compile("\\!\\[(.*)\\]\\((.*)\\)");
    Matcher matcher = compile.matcher(data);
    HashMap<String, String> images = new HashMap<>();
    while (matcher.find()) {
      String name = matcher.group(1);
      String url = matcher.group(2);
      images.put(name, url);
    }
    return images;
  }

  /**
   * Gets header json.
   *
   * @param data the data
   * @return the header json
   */
  public JsonObject getHeaderJson(String data) {
    Pattern compile = Pattern.compile("^\\{.*?\\}\\s*\r?\n", Pattern.DOTALL);
    Matcher matcher = compile.matcher(data);
    if (!matcher.find()) return new JsonObject();
    String trim = matcher.group(0).trim();
    return new GsonBuilder().setLenient().create().fromJson(trim, JsonObject.class);
  }

  /**
   * Sets header json.
   *
   * @param data      the data
   * @param newHeader the new header
   * @return the header json
   */
  public String setHeaderJson(String data, JsonObject newHeader) {
    Pattern compile = Pattern.compile("^(\\{.*\\})\n", Pattern.DOTALL);
    Matcher matcher = compile.matcher(data);
    String headerStr = new GsonBuilder().setLenient().setPrettyPrinting().create().toJson(newHeader);
    if (!matcher.find()) return headerStr + "\n" + data;
    return matcher.replaceFirst(headerStr + "\n");
  }

  /**
   * Download file.
   *
   * @param url      the url
   * @param imageDir the image dir
   * @return the file
   * @throws IOException the io exception
   */
  @Nullable
  public File download(String url, File imageDir) throws IOException {
    File file;
    if (url.startsWith(".") || url.startsWith("/")) {
      logger.warn("Local Link: " + url);
      file = null;
    } else {
      CloseableHttpClient build = HttpClientBuilder.create().build();
      CloseableHttpResponse response = build.execute(new HttpGet(url));
      String[] split = url.replaceAll("\\?.*", "").split("\\.");
      String extension;
      if (split.length == 0) {
        logger.warn("Unknown type of " + split[0] + " (assuming gif)");
        extension = "gif";
      } else {
        extension = split[split.length - 1];
        if (extension.length() > 4) {
          logger.warn("Unknown type of " + split[split.length - 1] + " (assuming gif)");
          extension = "gif";
        }
      }
      file = new File(imageDir, UUID.randomUUID().toString() + "." + extension);
      try (FileOutputStream outStream = new FileOutputStream(file)) {
        response.getEntity().writeTo(outStream);
      }
      logger.warn(String.format("Downloaded %s as %s", url, file));
    }
    return file;
  }

}
