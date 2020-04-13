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

import com.simiacryptus.aws.Tendril;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.util.test.NotebookReportBase;
import org.apache.commons.io.FileUtils;
import org.jetbrains.annotations.NotNull;
import org.junit.jupiter.api.Test;

import javax.annotation.Nonnull;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

public class BuildAndRelease extends NotebookReportBase {

  @Override
  public @Nonnull ReportType getReportType() {
    return ReportType.Components;
  }

  @Override
  protected Class<?> getTargetClass() {
    return BuildAndRelease.class;
  }

  @Test
  public void main() {
    //String buildDirectory = "/mnt/h/SimiaCryptus/all-projects";
    build(
        getLog(),
        TimeUnit.HOURS.toMillis(6),
        "C:\\Windows\\System32\\bash.exe",
        "git",
        "/mnt/c/Users/andre/Downloads/apache-maven-3.6.3-bin/apache-maven-3.6.3/bin/mvn",
        "H:\\SimiaCryptus\\all-projects", false
    );
  }

  public static void build(NotebookOutput log, long timeout, String bash, String git, String maven, String buildDirectory, boolean clean) {
    long endTime = System.currentTimeMillis() + TimeUnit.MINUTES.toMillis(50);
    try {
      String mainProject = "all-projects";
      String mainBuildDirectory = buildDirectory + "/" + mainProject;
      if (clean) {
        log.subreport("Git Checkout", sub -> {
          new File(buildDirectory).mkdirs();
          commands(sub, timeout, buildDirectory,
              new String[]{bash, "rm", "-rf", buildDirectory + "/" + mainProject},
              new String[]{git, "clone", "git@github.com:SimiaCryptus/" + mainProject + ".git"}
          );
//          commands(sub, timeout, mainBuildDirectory,
//              new String[]{bash, "./gitall.sh", "reset", "--hard"},
//              new String[]{bash, "./gitall.sh", "clean", "-fdx"},
//              new String[]{bash, git, "reset", "--hard"},
//              new String[]{bash, git, "clean", "-fdx"}
//          );
          commands(sub, timeout, mainBuildDirectory,
//              new String[]{bash, git, "pull", "origin", "master"},
              new String[]{git, "submodule", "update", "--init"}
          );
          return null;
        });
      }
      String newVersion = "2.0.0";
      HashMap<File, String> previousData = log.subreport("Updating Version to " + newVersion, sub -> {
        return setVersion(sub, mainBuildDirectory, newVersion, "MINDSEYE-SNAPSHOT");
      });
      log.h1("Building Build Tools");
      commands(log, timeout, mainBuildDirectory + "/third-party/aws-s3-maven",
          new String[]{bash, maven, "clean", "package", "install", "-fae", "-DskipTests"}
      );
      log.h1("Building Software");
      commands(log, timeout, mainBuildDirectory,
          new String[]{bash, maven, "clean", "package", "install", "-fae", "-DskipTests"}
      );
      log.h1("Building Site");
      commands(log, timeout, mainBuildDirectory,
          new String[]{bash, maven, "site:site", "-fae", "-Prelease", "-DskipTests"},
          new String[]{bash, maven,
              //"-s", "/mnt/c/Users/andre/.m2/settings.xml",
              "site:deploy", "-fae", "-Prelease", "-DskipTests"
          });
      log.h1("Deploy Software");
      commands(log, timeout, mainBuildDirectory,
          new String[]{bash, maven, "clean", "package", "deploy", "-fae", "-Prelease", "-DskipTests"}
      );
      log.subreport("Revert Version Changes", sub -> {
        revert(sub, mainBuildDirectory, previousData);
        return null;
      });
    } finally {
      while (System.currentTimeMillis() < endTime) {
        try {
          System.out.println("Waiting for total run time of 50 minutes...");
          Thread.sleep(TimeUnit.MINUTES.toMillis(1));
        } catch (InterruptedException e) {
          break;
        }
      }
    }
  }

  public static void revert(NotebookOutput log, String buildDirectory, HashMap<File, String> previousData) {
    for (File file : FileUtils.listFiles(new File(buildDirectory), new String[]{"xml"}, true)) {
      if (file.getName().equals("pom.xml")) {
        try {
          String originalData = FileUtils.readFileToString(file, "UTF-8");
          String newData = previousData.getOrDefault(file.getAbsoluteFile(), originalData);
          if (!newData.equals(originalData)) {
            log.p("Reverting " + file.getAbsolutePath());
            FileUtils.write(file, newData, "UTF-8");
          }
        } catch (IOException e) {
          e.printStackTrace();
        }
      }
    }
  }

  @NotNull
  public static HashMap<File, String> setVersion(NotebookOutput log, String buildDirectory, final String newVersion, final String oldVersion) {
    HashMap<File, String> previousData = new HashMap<>();
    for (File file : FileUtils.listFiles(new File(buildDirectory), new String[]{"xml"}, true)) {
      if (file.getName().equals("pom.xml")) {
        try {
          String originalData = FileUtils.readFileToString(file, "UTF-8");
          String newData = originalData.replaceAll("<version>" + oldVersion + "</version>", "<version>" + newVersion + "</version>");
          if (!newData.equals(originalData)) {
            previousData.put(file.getAbsoluteFile(), originalData);
            log.p("Modified " + file.getAbsolutePath());
            FileUtils.write(file, newData, "UTF-8");
          }
        } catch (IOException e) {
          e.printStackTrace();
        }
      }
    }
    return previousData;
  }

  public static void commands(NotebookOutput log, long timeout, String buildDirectory, String[]... commands) {
    for (String[] command : commands) {
      log.h2(Arrays.stream(command).filter(x -> x != null).map(x -> x.toString()).reduce((a, b) -> a + " " + b).get());
      try {
        ProcessBuilder processBuilder = new ProcessBuilder();
        processBuilder.command(Arrays.stream(command).filter(x -> x != null).collect(Collectors.toList()));
        processBuilder.directory(new File(buildDirectory));
        log.eval(() -> {
          return run(processBuilder, timeout).exitValue();
        });
      } catch (Throwable e) {
        e.printStackTrace();
      }
    }
  }

  public static Process run(ProcessBuilder processBuilder, long timeout) throws IOException, InterruptedException {
    Process process = processBuilder.start();
    try {
      Tendril.pump(process);
      long maxTime = System.currentTimeMillis() + timeout;
      while (process.isAlive() && System.currentTimeMillis() < maxTime) {
        Thread.sleep(1000);
      }
      return process;
    } finally {
      if (process.isAlive()) {
        process.destroy();
      }
    }
  }

}