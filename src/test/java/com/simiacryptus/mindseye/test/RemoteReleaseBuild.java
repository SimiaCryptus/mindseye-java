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

import com.simiacryptus.aws.exe.EC2NodeSettings;
import com.simiacryptus.aws.exe.EC2NotebookRunner;

import java.util.concurrent.TimeUnit;

/**
 * The type Remote build.
 */
public class RemoteReleaseBuild {

  /**
   * The entry point of application.
   *
   * @param args the input arguments
   */
  public static void main(String[] args) {
    EC2NotebookRunner.launch(
        EC2NodeSettings.M5_L,
        EC2NodeSettings.AMI_BUILD,
        " -Xmx8g -DTEST_REPO=./runner/",
        log -> {
          BuildAndRelease.build(
              log,
              TimeUnit.MINUTES.toMillis(240),
              "/bin/bash",
              "/usr/bin/git",
              "/home/ec2-user/apache-maven-3.6.3/bin/mvn",
              "/home/ec2-user/code",
              true,
              true,
              true,
              false,
              "code.simiacrypt.us/release",
              "MINDSEYE-SNAPSHOT",
              "2.0.1"
          );
        }
    );
  }

}
