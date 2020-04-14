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
import org.apache.commons.io.filefilter.IOFileFilter;
import org.jetbrains.annotations.NotNull;
import org.junit.jupiter.api.Test;

import javax.annotation.Nonnull;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

/**
 * The type Build and release.
 */
public class BuildAndRelease extends NotebookReportBase {

  @Override
  public @Nonnull ReportType getReportType() {
    return ReportType.Components;
  }

  @Override
  protected Class<?> getTargetClass() {
    return BuildAndRelease.class;
  }

  /**
   * Local build.
   */
  @Test
  public void localBuild() {
    //String buildDirectory = "/mnt/h/SimiaCryptus/all-projects";
    build(
        getLog(),
        TimeUnit.HOURS.toMillis(6),
        "C:\\Windows\\System32\\bash.exe",
        "git",
        "/mnt/c/Users/andre/Downloads/apache-maven-3.6.3-bin/apache-maven-3.6.3/bin/mvn",
        "H:\\SimiaCryptus", false, true, true, false
    );
  }

  /**
   * Standard site xml.
   */
  @Test
  public void standardSiteXML() {
    Collection<File> pomFiles = FileUtils.listFiles(new File("H:\\SimiaCryptus\\all-projects"), new IOFileFilter() {
      @Override
      public boolean accept(File file) {
        return "pom.xml".equals(file.getName());
      }

      @Override
      public boolean accept(File dir, String name) {
        return "pom.xml".equals(name);
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
    File modelSiteXml = new File("src/site/site.xml");
    pomFiles.forEach(pomFile -> {
      if (pomFile.toPath().resolve("../src").toFile().exists()) {
        File siteXml = pomFile.toPath().resolve("../src/site/site.xml").normalize().toFile();
        if (siteXml.exists()) {
          System.out.println(siteXml.getAbsolutePath() + " already exists");
        } else {
          try {
            FileUtils.copyFile(modelSiteXml, siteXml);
          } catch (IOException e) {
            e.printStackTrace();
          }
        }
        try {
          //String data = FileUtils.readFileToString(siteXml, "UTF-8");
          String data = FileUtils.readFileToString(modelSiteXml, "UTF-8");
          String name = pomFile.toPath().getParent().normalize().toAbsolutePath().toFile().getName();
          data = data.replaceAll("<project name=\".*\">", "<project name=\"" + name + "\">");
          data = data.replaceAll("<projectId>.*</projectId>", "<projectId>SimiaCryptus/" + name + "</projectId>");
          FileUtils.writeStringToFile(siteXml, data, "UTF-8");
        } catch (IOException e) {
          e.printStackTrace();
        }
      }
    });
  }


  /**
   * Build.
   *
   * @param log            the log
   * @param timeout        the timeout
   * @param bash           the bash
   * @param git            the git
   * @param maven          the maven
   * @param buildDirectory the build directory
   * @param clean          the clean
   * @param release
   * @param site
   * @param installTools
   */
  public static void build(NotebookOutput log, long timeout, String bash, String git, String maven, String buildDirectory, boolean clean, boolean release, boolean site, boolean installTools) {
    long endTime = System.currentTimeMillis() + TimeUnit.MINUTES.toMillis(50);
    try {
      if(installTools) {
        log.subreport("Tooling Setup", sub -> {
          commands(sub, timeout, buildDirectory,
              new String[]{bash, "-c", "sudo yum update"},
              new String[]{bash, "-c", "sudo yum install git default-jdk"},
              new String[]{bash, "-c", "rm -rf apache-maven-3.6.3*"},
              new String[]{bash, "-c", "wget http://apache.mirrors.hoobly.com/maven/maven-3/3.6.3/binaries/apache-maven-3.6.3-bin.tar.gz"},
              new String[]{bash, "-c", "tar -xzvf apache-maven-3.6.3-bin.tar.gz"}
          );
          return null;
        });
        maven = buildDirectory + "/apache-maven-3.6.3/bin/mvn/";
      }
      String mainProject = "all-projects";
      String mainBuildDirectory = buildDirectory + "/" + mainProject;
      if (clean) {
        log.subreport("Git Checkout", sub -> {
          new File(buildDirectory).mkdirs();
          commands(sub, timeout, buildDirectory,
              new String[]{bash, "-c rm -rf " + buildDirectory + "/" + mainProject},
              new String[]{git, "clone", "https://github.com/SimiaCryptus/" + mainProject + ".git"}
          );
          commands(sub, timeout, mainBuildDirectory,
              new String[]{git, "pull", "origin", "master"},
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
      log.h1("Validating Software Integrity");
      for(String dir : Arrays.asList(
          mainBuildDirectory + "/util",
          mainBuildDirectory + "/mindseye",
          mainBuildDirectory + "/misc",
          mainBuildDirectory + "/deepartist"
      )) {
        log.h2(dir);
        run(log, timeout, dir,
            new String[]{bash, maven, "clean", "com.simiacryptus:refcount-autocoder:verify", "-fae", "-DskipTests"}
        );
      }
      String profile = release ?"-Prelease":null;
      if(site) {
        log.h1("Building Site");
        commands(log, timeout, mainBuildDirectory,
            new String[]{bash, maven, "site:site", "-fae", profile, "-DskipTests"},
            new String[]{bash, maven, "site:stage", "-fae", profile, "-DskipTests"},
            new String[]{bash, maven, "site:deploy", "-fae", profile, "-DskipTests"}
        );
      }
      if(release) {
        log.h1("Deploy Software");
        commands(log, timeout, mainBuildDirectory,
            new String[]{bash, maven, "clean", "package", "deploy", "-fae", profile, "-DskipTests"}
        );
      }
      log.subreport("Revert Version Changes", sub -> {
        revert(sub, mainBuildDirectory, previousData);
        return null;
      });
    } finally {
      while (clean && System.currentTimeMillis() < endTime) {
        try {
          System.out.println("Waiting for total run time of 50 minutes...");
          Thread.sleep(TimeUnit.MINUTES.toMillis(1));
        } catch (InterruptedException e) {
          break;
        }
      }
    }
  }

  /**
   * Revert.
   *
   * @param log            the log
   * @param buildDirectory the build directory
   * @param previousData   the previous data
   */
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

  /**
   * Sets version.
   *
   * @param log            the log
   * @param buildDirectory the build directory
   * @param newVersion     the new version
   * @param oldVersion     the old version
   * @return the version
   */
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

  /**
   * Commands.
   *
   * @param log            the log
   * @param timeout        the timeout
   * @param buildDirectory the build directory
   * @param commands       the commands
   */
  public static void commands(NotebookOutput log, long timeout, String buildDirectory, String[]... commands) {
    new File(buildDirectory).mkdirs();
    for (String[] command : commands) {
      log.h2(Arrays.stream(command).filter(x -> x != null).map(x -> x.toString()).reduce((a, b) -> a + " " + b).get());
      run(log, timeout, buildDirectory, command);
    }
  }

  /**
   * Run.
   *
   * @param log            the log
   * @param timeout        the timeout
   * @param buildDirectory the build directory
   * @param command        the command
   */
  public static void run(NotebookOutput log, long timeout, String buildDirectory, String[] command) {
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

  /**
   * Run process.
   *
   * @param processBuilder the process builder
   * @param timeout        the timeout
   * @return the process
   * @throws IOException          the io exception
   * @throws InterruptedException the interrupted exception
   */
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
