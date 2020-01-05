/*
 * Copyright (c) 2019 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.layers.java;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.ref.lang.RefAware;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;

@SuppressWarnings("serial")
public @RefAware
class SignReducerLayer extends DAGNetwork {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(SignReducerLayer.class);
  private final DAGNode head;

  public SignReducerLayer() {
    super(1);
    final DAGNode avgInput = add(new AvgReducerLayer(), getInput(0));
    {
      SigmoidActivationLayer temp_01_0003 = new SigmoidActivationLayer();
      NthPowerActivationLayer temp_01_0004 = new NthPowerActivationLayer();
      NthPowerActivationLayer temp_01_0005 = new NthPowerActivationLayer();
      LinearActivationLayer temp_01_0006 = new LinearActivationLayer();
      DAGNode temp_01_0001 = add(temp_01_0003.setBalanced(false),
          add(new ProductInputsLayer(), avgInput == null ? null : avgInput.addRef(),
              add(temp_01_0004.setPower(-1),
                  add(temp_01_0005.setPower(0.5),
                      add(new SumInputsLayer(), add(new AvgReducerLayer(), add(new SqActivationLayer(), getInput(0))),
                          add(temp_01_0006.setScale(-1),
                              add(new SqActivationLayer(), avgInput == null ? null : avgInput.addRef())))))));
      if (null != temp_01_0006)
        temp_01_0006.freeRef();
      if (null != temp_01_0005)
        temp_01_0005.freeRef();
      if (null != temp_01_0004)
        temp_01_0004.freeRef();
      if (null != temp_01_0003)
        temp_01_0003.freeRef();
      head = temp_01_0001 == null ? null : temp_01_0001.addRef();
      if (null != temp_01_0001)
        temp_01_0001.freeRef();
    }
    if (null != avgInput)
      avgInput.freeRef();
  }

  protected SignReducerLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json, rs);
    {
      DAGNode temp_01_0002 = getNodeById(
          UUID.fromString(json.getAsJsonPrimitive("head").getAsString()));
      head = temp_01_0002 == null ? null : temp_01_0002.addRef();
      if (null != temp_01_0002)
        temp_01_0002.freeRef();
    }
  }

  @Override
  public DAGNode getHead() {
    return head == null ? null : head.addRef();
  }

  @SuppressWarnings("unused")
  public static Layer fromJson(@Nonnull final JsonObject inner, Map<CharSequence, byte[]> rs) {
    return new SignReducerLayer(inner, rs);
  }

  public static @SuppressWarnings("unused")
  SignReducerLayer[] addRefs(SignReducerLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(SignReducerLayer::addRef)
        .toArray((x) -> new SignReducerLayer[x]);
  }

  public static @SuppressWarnings("unused")
  SignReducerLayer[][] addRefs(SignReducerLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(SignReducerLayer::addRefs)
        .toArray((x) -> new SignReducerLayer[x][]);
  }

  public void _free() {
    if (null != head)
      head.freeRef();
    super._free();
  }

  public @Override
  @SuppressWarnings("unused")
  SignReducerLayer addRef() {
    return (SignReducerLayer) super.addRef();
  }
}
