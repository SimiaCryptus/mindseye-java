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
import com.simiacryptus.util.test.MacroTestRunner;
import com.simiacryptus.util.test.MacroTestRunner.Isolation;
import com.simiacryptus.util.CodeUtil;

public class RemoteTests {

  public static void main(String[] args) {
    EC2NotebookRunner.launch(
        EC2NodeSettings.M5_L,
        EC2NodeSettings.AMI_AMAZON_LINUX,
        " -Xmx16g -DTEST_REPO=./runner/",
        log -> {
          MacroTestRunner.runAll(log,
              "com.simiacryptus.mindseye.layers",
              Isolation.Class
          );
        }
    );
  }

}
