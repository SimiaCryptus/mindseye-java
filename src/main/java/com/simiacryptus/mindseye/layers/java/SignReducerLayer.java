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
    head = add(new SigmoidActivationLayer().setBalanced(false),
        add(new ProductInputsLayer(), avgInput,
            add(new NthPowerActivationLayer().setPower(-1),
                (DAGNode) add(new NthPowerActivationLayer().setPower(0.5),
                    add(new SumInputsLayer(), add(new AvgReducerLayer(), add(new SqActivationLayer(), getInput(0))),
                        add(new LinearActivationLayer().setScale(-1), add(new SqActivationLayer(), avgInput)))))));
  }

  protected SignReducerLayer(@Nonnull final JsonObject json,
                             Map<CharSequence, byte[]> rs) {
    super(json, rs);
    head = getNodeById(UUID.fromString(json.getAsJsonPrimitive("head").getAsString()));
  }

  @Override
  public DAGNode getHead() {
    return head;
  }

  @SuppressWarnings("unused")
  public static Layer fromJson(@Nonnull final JsonObject inner,
                               Map<CharSequence, byte[]> rs) {
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
    super._free();
  }

  public @Override
  @SuppressWarnings("unused")
  SignReducerLayer addRef() {
    return (SignReducerLayer) super.addRef();
  }
}
