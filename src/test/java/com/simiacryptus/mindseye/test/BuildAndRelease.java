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
import com.simiacryptus.util.Util;
import com.simiacryptus.util.test.NotebookTestBase;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.filefilter.IOFileFilter;
import org.jetbrains.annotations.NotNull;
import org.junit.jupiter.api.Test;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

import javax.annotation.Nonnull;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.transform.OutputKeys;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerException;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;
import javax.xml.xpath.XPath;
import javax.xml.xpath.XPathConstants;
import javax.xml.xpath.XPathExpressionException;
import javax.xml.xpath.XPathFactory;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.nio.file.FileVisitOption;
import java.nio.file.Files;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.function.BiConsumer;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * The BuildAndRelease class contains a static XPath variable that can be used to instantiate new XPath objects.
 *
 * @docgenVersion 9
 */
public class BuildAndRelease extends NotebookTestBase {

  private final static XPath xPath = XPathFactory.newInstance().newXPath();

  @Override
  public @Nonnull ReportType getReportType() {
    return ReportType.Components;
  }

  @Override
  protected Class<?> getTargetClass() {
    return BuildAndRelease.class;
  }

  /**
   * Updates the POMs in the given root directory.
   *
   * @param root the root directory
   * @param fn   the function to apply to each POM
   * @docgenVersion 9
   */
  public static void updatePOMs(File root, BiConsumer<File, Document> fn) {
    try {
      List<File> poms = Files.walk(root.toPath(), FileVisitOption.FOLLOW_LINKS)
          .filter(x -> x.getFileName().toString().equals("pom.xml"))
          .map(x -> x.toFile())
          .collect(Collectors.toList());
      poms.forEach(pomFile -> {
        try {
          System.out.println("Editing " + pomFile.getAbsolutePath());
          Document xmlDoc = open(pomFile);
          backup(pomFile);
          fn.accept(pomFile, xmlDoc);
          write(pomFile, xmlDoc);
        } catch (Throwable e) {
          e.printStackTrace();
        }
      });
    } catch (Throwable e) {
      throw Util.throwException(e);
    }
  }

  /**
   * Open a pom file.
   *
   * @param pomFile the pom file to open
   * @return the document representing the pom file
   * @throws SAXException                 if there is a problem parsing the pom file
   * @throws IOException                  if there is a problem reading the pom file
   * @throws ParserConfigurationException if there is a problem configuring the parser
   * @docgenVersion 9
   */
  public static Document open(File pomFile) throws SAXException, IOException, ParserConfigurationException {
    return DocumentBuilderFactory.newInstance().newDocumentBuilder().parse(pomFile);
  }

  /**
   * This method backs up a file by copying it to a file with the same name plus the ".bak" extension.
   *
   * @param pomFile the file to back up
   * @throws IOException if there is an error copying the file
   * @docgenVersion 9
   */
  public static void backup(File pomFile) throws IOException {
    FileUtils.copyFile(pomFile, new File(pomFile.toString() + ".bak"));
  }

  /**
   * Edits the given XML document to match the given version.
   *
   * @param version The version to match.
   * @param xmlDoc  The XML document to edit.
   * @throws XPathExpressionException If an error occurs while editing the document.
   * @docgenVersion 9
   */
  public static void edit(String version, Document xmlDoc) throws XPathExpressionException {
    XPath xPath = XPathFactory.newInstance().newXPath();
    Node artifactId = (Node) xPath.evaluate("/project/artifactId", xmlDoc, XPathConstants.NODE);
    String projectName = artifactId.getTextContent();
    System.out.println("Project " + projectName);
    String targetUrl = "http://code.simiacrypt.us/release/" + version + "/" + projectName;
    Node url = (Node) xPath.compile("/project/url").evaluate(xmlDoc, XPathConstants.NODE);
    if (null == url) {
      System.out.println("No URL");
      Node element = getOrCreate(artifactId.getParentNode(), "url");
      element.setTextContent(targetUrl);
      Node parentNode = artifactId.getParentNode();
      parentNode.appendChild(element);
      //parentNode.insertBefore(project, element);
    } else {
      System.out.println("URL changed from " + url.getTextContent());
      url.setTextContent(targetUrl);
    }
  }

  /**
   * @param pomFile
   * @param xmlDoc
   * @throws TransformerException
   * @docgenVersion 9
   */
  public static void write(File pomFile, Document xmlDoc) throws TransformerException {
    Transformer transformer = TransformerFactory.newInstance().newTransformer();
    transformer.setOutputProperty(OutputKeys.INDENT, "yes");
    transformer.setOutputProperty(OutputKeys.METHOD, "xml");
    transformer.setOutputProperty("{http://xml.apache.org/xslt}indent-amount", "4");

    DOMSource domSource = new DOMSource(xmlDoc);
    StreamResult sr = new StreamResult(pomFile);
    transformer.transform(domSource, sr);
  }

  /**
   * Returns a string representation of an XML document.
   *
   * @param xmlDoc the XML document to convert to a string
   * @return a string representation of the XML document
   * @throws TransformerException         if an error occurs during conversion
   * @throws UnsupportedEncodingException if the document uses an unsupported encoding
   * @docgenVersion 9
   */
  public static String toString(Node xmlDoc) throws TransformerException, UnsupportedEncodingException {
    Transformer transformer = TransformerFactory.newInstance().newTransformer();
    transformer.setOutputProperty(OutputKeys.INDENT, "yes");
    transformer.setOutputProperty(OutputKeys.METHOD, "xml");
    transformer.setOutputProperty("{http://xml.apache.org/xslt}indent-amount", "4");

    DOMSource domSource = new DOMSource(xmlDoc);
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    StreamResult sr = new StreamResult(out);
    transformer.transform(domSource, sr);
    return out.toString("UTF-8");
  }

  /**
   * Build the project using the given tools.
   *
   * @param log            the log to use
   * @param timeout        the timeout in milliseconds
   * @param bash           the path to the bash executable
   * @param git            the path to the git executable
   * @param maven          the path to the maven executable
   * @param buildDirectory the build directory
   * @param clean          whether to clean the build directory before building
   * @param release        whether to do a release build
   * @param site           whether to generate a site
   * @param push           whether to push changes to the remote repository
   * @param installTools   whether to install build tools
   * @param siteHome       the directory to use for the generated site
   * @param branch         the branch to build
   * @param merge          the branch to merge with
   * @param oldVersion     the old version
   * @param newVersion     the new version
   * @docgenVersion 9
   */
  public static void build(NotebookOutput log, long timeout, String bash, String git, String maven, String buildDirectory, boolean clean, boolean release, boolean site, boolean push, boolean installTools, String siteHome, String branch, String merge, String oldVersion, String newVersion) {
    String mainProject = "all-projects";
    long endTime = System.currentTimeMillis() + TimeUnit.MINUTES.toMillis(15);
    try {
      if (installTools) {
        log.subreport("Tooling Setup", sub -> {
          commands(sub, timeout, buildDirectory,
              //new String[]{bash, "-c", "sudo yum update"},
              //new String[]{bash, "-c", "sudo yum install git default-jdk"},
              new String[]{bash, "-c", "java -version"},
              new String[]{bash, "-c", "javac -version"},
              new String[]{bash, "-c", "rm -rf apache-maven-3.6.3*"},
              new String[]{bash, "-c", "wget http://apache.mirrors.hoobly.com/maven/maven-3/3.6.3/binaries/apache-maven-3.6.3-bin.tar.gz"},
              new String[]{bash, "-c", "tar -xzvf apache-maven-3.6.3-bin.tar.gz"}
          );
          return null;
        });
        maven = buildDirectory + "/apache-maven-3.6.3/bin/mvn";
      }
      String mainBuildDirectory = buildDirectory + "/" + mainProject;
      if (clean) {
        log.subreport("Git Checkout", sub -> {
          new File(buildDirectory).mkdirs();
          commands(sub, timeout, buildDirectory,
              new String[]{bash, "-c rm -rf " + buildDirectory + "/" + mainProject},
              new String[]{git, "clone", "https://github.com/SimiaCryptus/" + mainProject + ".git"}
          );
          commands(sub, timeout, mainBuildDirectory,
              new String[]{git, "fetch", "origin"},
              new String[]{"./gitall.sh", "fetch", "origin"},
              new String[]{git, "checkout", branch},
              new String[]{git, "pull", "origin", branch},
              new String[]{git, "submodule", "update", "--init"}
          );
          if (null != merge && !merge.isEmpty()) {
            commands(sub, timeout, mainBuildDirectory,
                new String[]{"./gitall.sh", "merge", merge}
            );
          }
          return null;
        });
      }
      HashMap<File, String> previousData = log.subreport("Updating Version to " + newVersion, sub -> {
        return newVersion.equals(oldVersion) ? null : setVersion(sub, mainBuildDirectory, newVersion, oldVersion, siteHome);
      });
      if (push) {
        log.subreport("Git Push", sub -> {
          commands(sub, timeout, mainBuildDirectory,
              new String[]{"./commit.sh", "Release " + newVersion},
              new String[]{"./gitall.sh", "tag", newVersion},
              new String[]{git, "push", newVersion},
              new String[]{"./gitall.sh", "push", branch},
              new String[]{git, "push", branch}
          );
          return null;
        });
      }
      log.h1("Building Build Tools");
      commands(log, timeout, mainBuildDirectory + "/third-party/aws-s3-maven",
          new String[]{bash, maven, "clean", "package", "install", "-fae", "-DskipTests"}
      );
      log.h1("Building Software");
      commands(log, timeout, mainBuildDirectory,
          new String[]{bash, maven, "clean", "package", "install", "-fae", "-DskipTests"}
      );
      log.h1("Validating Software Integrity");
      for (String dir : Arrays.asList(
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
      String profile = release ? "-Prelease" : null;
      if (site) {
        log.h1("Building Site");
        commands(log, timeout, mainBuildDirectory,
            new String[]{bash, maven, "site:site", "-fae", profile, "-DskipTests"},
            new String[]{bash, maven, "site:stage", "-fae", profile, "-DskipTests"},
            new String[]{bash, maven, "site:deploy", "-fae", profile, "-DskipTests"}
        );
      }
      if (release) {
        log.h1("Deploy Software");
        commands(log, timeout, mainBuildDirectory,
            new String[]{bash, maven, "clean", "package", "deploy", "-fae", profile, "-DskipTests"}
        );
      } else {
        log.h1("Build and Install Software");
        commands(log, timeout, mainBuildDirectory,
            new String[]{bash, maven, "clean", "package", "install", "-fae", profile, "-DskipTests"}
        );
      }
      if (null != previousData) log.subreport("Revert Version Changes", sub -> {
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
   * Reverts the given log to the state it was in when the given build directory was last built.
   *
   * @param log            the log to revert
   * @param buildDirectory the build directory to revert to
   * @param previousData   a map of files to their data at the time of the build
   * @docgenVersion 9
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
   * Sets the version of the site.
   *
   * @param log            the log to write to
   * @param buildDirectory the directory to build in
   * @param newVersion     the new version
   * @param oldVersion     the old version
   * @param siteHome       the home directory of the site
   * @return a map of files to their new versions
   * @docgenVersion 9
   */
  @NotNull
  public static HashMap<File, String> setVersion(NotebookOutput log, String buildDirectory, final String newVersion, final String oldVersion, String siteHome) {
    HashMap<File, String> previousData = new HashMap<>();
    for (File file : FileUtils.listFiles(new File(buildDirectory), new String[]{"xml"}, true)) {
      if (file.getName().equals("pom.xml")) {
        try {
          String originalData = FileUtils.readFileToString(file, "UTF-8");
          String newData = originalData
              .replaceAll("<version>" + oldVersion + "</version>", "<version>" + newVersion + "</version>")
              .replaceAll("code.simiacrypt.us/release", siteHome);
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
   * @param log            the log to write to
   * @param timeout        the timeout in milliseconds
   * @param buildDirectory the build directory
   * @param commands       the commands to run
   * @docgenVersion 9
   */
  public static void commands(NotebookOutput log, long timeout, String buildDirectory, String[]... commands) {
    new File(buildDirectory).mkdirs();
    for (String[] command : commands) {
      log.h2(Arrays.stream(command).filter(x -> x != null).map(x -> x.toString()).reduce((a, b) -> a + " " + b).get());
      run(log, timeout, buildDirectory, command);
    }
  }

  /**
   * @param log            the log to write to
   * @param timeout        the timeout in milliseconds
   * @param buildDirectory the build directory
   * @param command        the command to run
   * @docgenVersion 9
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
   * Runs a process with a given timeout.
   *
   * @param processBuilder the process to run
   * @param timeout        the timeout in milliseconds
   * @return the running process
   * @throws IOException          if an I/O error occurs
   * @throws InterruptedException if the process is interrupted
   * @docgenVersion 9
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

  /**
   * Returns the node with the given name, or creates a new node with that name if one does not exist.
   *
   * @param node  the starting node
   * @param names the names of the nodes to find or create
   * @return the node with the given name
   * @docgenVersion 9
   */
  public static Node getOrCreate(Node node, String... names) {
    String name = names[0];
    NodeList childNodes = node.getChildNodes();
    for (int i = 0; i < childNodes.getLength(); i++) {
      Node item = childNodes.item(i);
      if (item.getNodeName().equals(name)) {
        if (names.length > 1) return getOrCreate(item, Arrays.copyOfRange(names, 1, names.length));
        return item;
      }
    }
    Element element = node.getOwnerDocument().createElement(name);
    node.appendChild(element);
    if (names.length > 1) return getOrCreate(element, Arrays.copyOfRange(names, 1, names.length));
    return element;
  }

  /**
   * Returns the node with the given name.
   *
   * @param node  the node to search
   * @param names the name of the node to find
   * @return the node with the given name
   * @docgenVersion 9
   */
  public static Node get(Node node, String... names) {
    String name = names[0];
    NodeList childNodes = node.getChildNodes();
    for (int i = 0; i < childNodes.getLength(); i++) {
      Node item = childNodes.item(i);
      if (item.getNodeName().equals(name)) {
        if (names.length > 1) return get(item, Arrays.copyOfRange(names, 1, names.length));
        return item;
      }
    }
    return null;
  }

  /**
   * Test to see if the build is local.
   *
   * @docgenVersion 9
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
        "H:\\SimiaCryptus",
        false, false, false, false, false,
        "code.simiacrypt.us/release",
        "master",
        "origin/develop",
        "MINDSEYE-SNAPSHOT",
        "2.0.0"
    );
  }

  /**
   * @Test public void updatePOMs();
   * <p>
   * This method will update the POMs for the project.
   * @docgenVersion 9
   */
  @Test
  public void updatePOMs() {
    List<Node> versionedDeps = new ArrayList<>();
    updatePOMs(new File(new File("H:\\SimiaCryptus"), "all-projects"), (File pomFile, Document document) -> {
      try {
        if (pomFile.getAbsolutePath().contains("third-party")) return;
        Node artifactId = (Node) xPath.evaluate("/project/artifactId", document, XPathConstants.NODE);
        System.out.println("Project " + artifactId.getTextContent());

        {
          String targetUrl = "http://code.simiacrypt.us/release/${project.version}/" + artifactId.getTextContent();
          Node url = (Node) xPath.compile("/project/url").evaluate(document, XPathConstants.NODE);
          if (null == url) {
            System.out.println("No URL");
            Node element = getOrCreate(artifactId.getParentNode(), "url");
            element.setTextContent(targetUrl);
            artifactId.getParentNode().appendChild(element);
            //parentNode.insertBefore(project, element);
          } else {
            System.out.println("URL changed from " + url.getTextContent());
            url.setTextContent(targetUrl);
          }
        }

        {
          String targetUrl = "s3://code.simiacrypt.us/release/${project.version}/" + artifactId.getTextContent();
          Node url = (Node) xPath.compile("/project/distributionManagement/site/url").evaluate(document, XPathConstants.NODE);
          if (null == url) {
            System.out.println("No URL");
            getOrCreate(artifactId.getParentNode(), "distributionManagement", "site", "url").setTextContent(targetUrl);
            //parentNode.insertBefore(project, element);
          } else {
            System.out.println("URL changed from " + url.getTextContent());
            url.setTextContent(targetUrl);
          }
        }

        {
          Node dependencies = getOrCreate(artifactId.getParentNode(), "dependencyManagement", "dependencies");
          if (dependencies.getChildNodes().getLength() == 0) {
            Element dependency = dependencies.getOwnerDocument().createElement("dependency");
            dependencies.appendChild(dependency);
            getOrCreate(dependency, "groupId").setTextContent("com.simiacryptus");
            getOrCreate(dependency, "artifactId").setTextContent("bom");
            getOrCreate(dependency, "version").setTextContent("${project.version}");
            getOrCreate(dependency, "type").setTextContent("pom");
            getOrCreate(dependency, "scope").setTextContent("import");
          }
        }

        {
          Node dependencies = get(artifactId.getParentNode(), "dependencies");
          if (null != dependencies) {
            NodeList childNodes = dependencies.getChildNodes();
            versionedDeps.addAll(IntStream.range(0, childNodes.getLength()).mapToObj(i -> {
              Node dependency = childNodes.item(i);
              Node versionNode = get(dependency, "version");
              if (null != versionNode) {
                versionNode.getParentNode().removeChild(versionNode);
                return dependency;
              } else {
                return null;
              }
            }).filter(x -> x != null).collect(Collectors.toList()));
          }
        }
      } catch (Throwable e) {
        e.printStackTrace();
      }
    });
    try {
      Document bom = open(new File("H:\\SimiaCryptus\\all-projects\\mvn-parents\\bom\\pom.xml"));
      Node bom_dependencies = (Node) xPath.evaluate("/project/dependencyManagement/dependencies", bom, XPathConstants.NODE);
      NodeList nodes = bom_dependencies.getChildNodes();
      for (int i = 0; i < nodes.getLength(); i++) {
        Node item = nodes.item(i);
        versionedDeps.add(item);
        bom_dependencies.removeChild(item);
      }
      versionedDeps.stream()
          .collect(Collectors.groupingBy(node -> String.format("%s:%s", getText(node, "groupId"), getText(node, "artifactId"))))
          .values().stream()
          .filter(x -> !x.isEmpty())
          .map(x -> x.get(0))
          .sorted(Comparator.comparing(node -> String.format("%s:%s", getText(node, "groupId"), getText(node, "artifactId"))))
          .map(x -> bom_dependencies.getOwnerDocument().adoptNode(x))
          .collect(Collectors.toList())
          .forEach(bom_dependencies::appendChild);
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }

  /**
   * Returns the text content of the node with the given group ID.
   *
   * @param x1      the node to get the text content from
   * @param groupId the group ID of the node
   * @return the text content of the node, or an empty string if the node does not exist
   * @docgenVersion 9
   */
  public String getText(Node x1, String groupId) {
    Node node = get(x1, groupId);
    if (null == node) return "";
    return node.getTextContent();
  }

  /**
   * @Test public void standardSiteXML();
   * <p>
   * This test will check that the standardSite.xml file is valid.
   * @docgenVersion 9
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
}
