module     p2_gg_httbar_d66h0l1d_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d66h0l1d_qp.f90
   ! generator: buildfortran_d.py
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_util_qp, only: cond, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, private :: iv0
   integer, private :: iv1
   integer, private :: iv2
   integer, private :: iv3
   real(ki), dimension(4), private :: qshift
   public :: derivative
contains
!---#[ function brack_1:
   pure function brack_1(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd66h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(74) :: acd66
      complex(ki) :: brack
      acd66(1)=dotproduct(k1,qshift)
      acd66(2)=dotproduct(qshift,qshift)
      acd66(3)=abb66(38)
      acd66(4)=dotproduct(qshift,spval4k2)
      acd66(5)=abb66(48)
      acd66(6)=dotproduct(qshift,spval4l3)
      acd66(7)=abb66(71)
      acd66(8)=dotproduct(qshift,spval5k2)
      acd66(9)=abb66(63)
      acd66(10)=dotproduct(qshift,spval5l3)
      acd66(11)=abb66(22)
      acd66(12)=abb66(44)
      acd66(13)=dotproduct(k2,qshift)
      acd66(14)=abb66(19)
      acd66(15)=dotproduct(qshift,spvak1k2)
      acd66(16)=abb66(14)
      acd66(17)=abb66(49)
      acd66(18)=dotproduct(l3,qshift)
      acd66(19)=abb66(33)
      acd66(20)=abb66(12)
      acd66(21)=abb66(21)
      acd66(22)=dotproduct(qshift,spval4k1)
      acd66(23)=abb66(30)
      acd66(24)=dotproduct(qshift,spval5k1)
      acd66(25)=abb66(28)
      acd66(26)=abb66(32)
      acd66(27)=abb66(10)
      acd66(28)=abb66(43)
      acd66(29)=dotproduct(qshift,spvak2k1)
      acd66(30)=abb66(31)
      acd66(31)=abb66(18)
      acd66(32)=dotproduct(qshift,spval3k2)
      acd66(33)=dotproduct(qshift,spval3k1)
      acd66(34)=abb66(24)
      acd66(35)=abb66(9)
      acd66(36)=abb66(27)
      acd66(37)=abb66(17)
      acd66(38)=dotproduct(qshift,spvak1l3)
      acd66(39)=abb66(25)
      acd66(40)=abb66(37)
      acd66(41)=abb66(11)
      acd66(42)=dotproduct(qshift,spvak1l4)
      acd66(43)=abb66(45)
      acd66(44)=dotproduct(qshift,spvak1l5)
      acd66(45)=abb66(42)
      acd66(46)=abb66(40)
      acd66(47)=dotproduct(qshift,spvak2l3)
      acd66(48)=abb66(26)
      acd66(49)=dotproduct(qshift,spvak2l5)
      acd66(50)=abb66(29)
      acd66(51)=abb66(16)
      acd66(52)=dotproduct(qshift,spval3l5)
      acd66(53)=abb66(20)
      acd66(54)=abb66(13)
      acd66(55)=-acd66(13)+acd66(1)
      acd66(55)=acd66(3)*acd66(55)
      acd66(56)=acd66(24)*acd66(25)
      acd66(57)=-acd66(22)*acd66(23)
      acd66(58)=-acd66(4)*acd66(20)
      acd66(59)=-acd66(8)*acd66(21)
      acd66(55)=acd66(59)+acd66(58)+acd66(57)+acd66(26)+acd66(56)+acd66(55)
      acd66(55)=acd66(2)*acd66(55)
      acd66(56)=acd66(6)*acd66(7)
      acd66(57)=acd66(4)*acd66(5)
      acd66(58)=acd66(10)*acd66(11)
      acd66(56)=-acd66(58)+acd66(56)-acd66(57)
      acd66(57)=acd66(8)*acd66(9)
      acd66(57)=acd66(57)+acd66(12)-acd66(56)
      acd66(57)=acd66(1)*acd66(57)
      acd66(58)=acd66(15)*acd66(16)
      acd66(59)=acd66(8)*acd66(14)
      acd66(56)=acd66(59)-acd66(17)+acd66(58)+acd66(56)
      acd66(56)=acd66(13)*acd66(56)
      acd66(58)=-acd66(25)*acd66(33)
      acd66(59)=acd66(32)*acd66(21)
      acd66(58)=acd66(59)-acd66(34)+acd66(58)
      acd66(58)=acd66(10)*acd66(58)
      acd66(59)=-acd66(52)*acd66(53)
      acd66(60)=-acd66(49)*acd66(50)
      acd66(61)=-acd66(47)*acd66(48)
      acd66(62)=-acd66(44)*acd66(45)
      acd66(63)=-acd66(42)*acd66(43)
      acd66(64)=-acd66(18)*acd66(19)
      acd66(65)=-acd66(38)*acd66(40)
      acd66(66)=-acd66(33)*acd66(51)
      acd66(67)=-acd66(29)*acd66(46)
      acd66(68)=-acd66(24)*acd66(37)
      acd66(69)=-acd66(22)*acd66(36)
      acd66(70)=-acd66(15)*acd66(35)
      acd66(71)=acd66(38)*acd66(39)
      acd66(71)=-acd66(41)+acd66(71)
      acd66(71)=acd66(32)*acd66(71)
      acd66(72)=-acd66(6)*acd66(28)
      acd66(73)=-acd66(4)*acd66(27)
      acd66(74)=-acd66(29)*acd66(30)
      acd66(74)=-acd66(31)+acd66(74)
      acd66(74)=acd66(8)*acd66(74)
      brack=acd66(54)+acd66(55)+acd66(56)+acd66(57)+acd66(58)+acd66(59)+acd66(6&
      &0)+acd66(61)+acd66(62)+acd66(63)+acd66(64)+acd66(65)+acd66(66)+acd66(67)&
      &+acd66(68)+acd66(69)+acd66(70)+acd66(71)+acd66(72)+acd66(73)+acd66(74)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd66h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(91) :: acd66
      complex(ki) :: brack
      acd66(1)=k1(iv1)
      acd66(2)=dotproduct(qshift,qshift)
      acd66(3)=abb66(38)
      acd66(4)=dotproduct(qshift,spval4k2)
      acd66(5)=abb66(48)
      acd66(6)=dotproduct(qshift,spval4l3)
      acd66(7)=abb66(71)
      acd66(8)=dotproduct(qshift,spval5k2)
      acd66(9)=abb66(63)
      acd66(10)=dotproduct(qshift,spval5l3)
      acd66(11)=abb66(22)
      acd66(12)=abb66(44)
      acd66(13)=k2(iv1)
      acd66(14)=abb66(19)
      acd66(15)=dotproduct(qshift,spvak1k2)
      acd66(16)=abb66(14)
      acd66(17)=abb66(49)
      acd66(18)=l3(iv1)
      acd66(19)=abb66(33)
      acd66(20)=qshift(iv1)
      acd66(21)=dotproduct(k1,qshift)
      acd66(22)=dotproduct(k2,qshift)
      acd66(23)=abb66(12)
      acd66(24)=abb66(21)
      acd66(25)=dotproduct(qshift,spval4k1)
      acd66(26)=abb66(30)
      acd66(27)=dotproduct(qshift,spval5k1)
      acd66(28)=abb66(28)
      acd66(29)=abb66(32)
      acd66(30)=spval4k2(iv1)
      acd66(31)=abb66(10)
      acd66(32)=spval4l3(iv1)
      acd66(33)=abb66(43)
      acd66(34)=spval5k2(iv1)
      acd66(35)=dotproduct(qshift,spvak2k1)
      acd66(36)=abb66(31)
      acd66(37)=abb66(18)
      acd66(38)=spval5l3(iv1)
      acd66(39)=dotproduct(qshift,spval3k2)
      acd66(40)=dotproduct(qshift,spval3k1)
      acd66(41)=abb66(24)
      acd66(42)=spvak1k2(iv1)
      acd66(43)=abb66(9)
      acd66(44)=spval4k1(iv1)
      acd66(45)=abb66(27)
      acd66(46)=spval5k1(iv1)
      acd66(47)=abb66(17)
      acd66(48)=spvak1l3(iv1)
      acd66(49)=abb66(25)
      acd66(50)=abb66(37)
      acd66(51)=spval3k2(iv1)
      acd66(52)=dotproduct(qshift,spvak1l3)
      acd66(53)=abb66(11)
      acd66(54)=spvak1l4(iv1)
      acd66(55)=abb66(45)
      acd66(56)=spvak1l5(iv1)
      acd66(57)=abb66(42)
      acd66(58)=spvak2k1(iv1)
      acd66(59)=abb66(40)
      acd66(60)=spvak2l3(iv1)
      acd66(61)=abb66(26)
      acd66(62)=spvak2l5(iv1)
      acd66(63)=abb66(29)
      acd66(64)=spval3k1(iv1)
      acd66(65)=abb66(16)
      acd66(66)=spval3l5(iv1)
      acd66(67)=abb66(20)
      acd66(68)=acd66(22)-acd66(21)
      acd66(68)=acd66(3)*acd66(68)
      acd66(69)=acd66(26)*acd66(25)
      acd66(70)=acd66(4)*acd66(23)
      acd66(71)=-acd66(28)*acd66(27)
      acd66(72)=acd66(8)*acd66(24)
      acd66(68)=acd66(72)+acd66(71)+acd66(70)-acd66(29)+acd66(69)+acd66(68)
      acd66(68)=acd66(20)*acd66(68)
      acd66(69)=acd66(26)*acd66(44)
      acd66(70)=acd66(30)*acd66(23)
      acd66(71)=-acd66(28)*acd66(46)
      acd66(72)=acd66(34)*acd66(24)
      acd66(69)=acd66(72)+acd66(71)+acd66(69)+acd66(70)
      acd66(69)=acd66(2)*acd66(69)
      acd66(70)=acd66(10)*acd66(11)
      acd66(71)=acd66(7)*acd66(6)
      acd66(72)=acd66(5)*acd66(4)
      acd66(73)=acd66(2)*acd66(3)
      acd66(70)=acd66(70)-acd66(71)+acd66(72)+acd66(73)
      acd66(71)=-acd66(8)*acd66(9)
      acd66(71)=acd66(71)-acd66(12)-acd66(70)
      acd66(71)=acd66(1)*acd66(71)
      acd66(72)=-acd66(16)*acd66(15)
      acd66(73)=-acd66(8)*acd66(14)
      acd66(70)=acd66(73)+acd66(17)+acd66(72)+acd66(70)
      acd66(70)=acd66(13)*acd66(70)
      acd66(72)=acd66(7)*acd66(32)
      acd66(73)=acd66(5)*acd66(30)
      acd66(74)=acd66(38)*acd66(11)
      acd66(72)=-acd66(74)+acd66(72)-acd66(73)
      acd66(73)=-acd66(34)*acd66(9)
      acd66(73)=acd66(73)+acd66(72)
      acd66(73)=acd66(21)*acd66(73)
      acd66(74)=-acd66(16)*acd66(42)
      acd66(75)=-acd66(34)*acd66(14)
      acd66(72)=acd66(75)+acd66(74)-acd66(72)
      acd66(72)=acd66(22)*acd66(72)
      acd66(74)=acd66(28)*acd66(64)
      acd66(75)=-acd66(24)*acd66(51)
      acd66(74)=acd66(74)+acd66(75)
      acd66(74)=acd66(10)*acd66(74)
      acd66(75)=acd66(28)*acd66(40)
      acd66(76)=-acd66(24)*acd66(39)
      acd66(75)=acd66(76)+acd66(41)+acd66(75)
      acd66(75)=acd66(38)*acd66(75)
      acd66(76)=acd66(8)*acd66(36)
      acd66(76)=acd66(76)+acd66(59)
      acd66(76)=acd66(58)*acd66(76)
      acd66(77)=-acd66(39)*acd66(49)
      acd66(77)=acd66(77)+acd66(50)
      acd66(77)=acd66(48)*acd66(77)
      acd66(78)=acd66(66)*acd66(67)
      acd66(79)=acd66(62)*acd66(63)
      acd66(80)=acd66(60)*acd66(61)
      acd66(81)=acd66(56)*acd66(57)
      acd66(82)=acd66(54)*acd66(55)
      acd66(83)=acd66(18)*acd66(19)
      acd66(84)=acd66(64)*acd66(65)
      acd66(85)=acd66(46)*acd66(47)
      acd66(86)=acd66(44)*acd66(45)
      acd66(87)=acd66(42)*acd66(43)
      acd66(88)=-acd66(49)*acd66(52)
      acd66(88)=acd66(53)+acd66(88)
      acd66(88)=acd66(51)*acd66(88)
      acd66(89)=acd66(32)*acd66(33)
      acd66(90)=acd66(30)*acd66(31)
      acd66(91)=acd66(36)*acd66(35)
      acd66(91)=acd66(37)+acd66(91)
      acd66(91)=acd66(34)*acd66(91)
      brack=2.0_ki*acd66(68)+acd66(69)+acd66(70)+acd66(71)+acd66(72)+acd66(73)+&
      &acd66(74)+acd66(75)+acd66(76)+acd66(77)+acd66(78)+acd66(79)+acd66(80)+ac&
      &d66(81)+acd66(82)+acd66(83)+acd66(84)+acd66(85)+acd66(86)+acd66(87)+acd6&
      &6(88)+acd66(89)+acd66(90)+acd66(91)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd66h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(59) :: acd66
      complex(ki) :: brack
      acd66(1)=d(iv1,iv2)
      acd66(2)=dotproduct(k1,qshift)
      acd66(3)=abb66(38)
      acd66(4)=dotproduct(k2,qshift)
      acd66(5)=dotproduct(qshift,spval4k1)
      acd66(6)=abb66(30)
      acd66(7)=dotproduct(qshift,spval4k2)
      acd66(8)=abb66(12)
      acd66(9)=dotproduct(qshift,spval5k1)
      acd66(10)=abb66(28)
      acd66(11)=dotproduct(qshift,spval5k2)
      acd66(12)=abb66(21)
      acd66(13)=abb66(32)
      acd66(14)=k1(iv1)
      acd66(15)=qshift(iv2)
      acd66(16)=spval4k2(iv2)
      acd66(17)=abb66(48)
      acd66(18)=spval5k2(iv2)
      acd66(19)=abb66(63)
      acd66(20)=spval4l3(iv2)
      acd66(21)=abb66(71)
      acd66(22)=spval5l3(iv2)
      acd66(23)=abb66(22)
      acd66(24)=k1(iv2)
      acd66(25)=qshift(iv1)
      acd66(26)=spval4k2(iv1)
      acd66(27)=spval5k2(iv1)
      acd66(28)=spval4l3(iv1)
      acd66(29)=spval5l3(iv1)
      acd66(30)=k2(iv1)
      acd66(31)=abb66(19)
      acd66(32)=spvak1k2(iv2)
      acd66(33)=abb66(14)
      acd66(34)=k2(iv2)
      acd66(35)=spvak1k2(iv1)
      acd66(36)=spval4k1(iv2)
      acd66(37)=spval5k1(iv2)
      acd66(38)=spval4k1(iv1)
      acd66(39)=spval5k1(iv1)
      acd66(40)=spvak2k1(iv2)
      acd66(41)=abb66(31)
      acd66(42)=spvak2k1(iv1)
      acd66(43)=spval3k2(iv2)
      acd66(44)=spval3k1(iv2)
      acd66(45)=spval3k2(iv1)
      acd66(46)=spval3k1(iv1)
      acd66(47)=spvak1l3(iv1)
      acd66(48)=abb66(25)
      acd66(49)=spvak1l3(iv2)
      acd66(50)=-acd66(8)*acd66(7)
      acd66(51)=-acd66(6)*acd66(5)
      acd66(52)=-acd66(12)*acd66(11)
      acd66(53)=acd66(10)*acd66(9)
      acd66(54)=-acd66(4)+acd66(2)
      acd66(54)=acd66(3)*acd66(54)
      acd66(50)=acd66(54)+acd66(53)+acd66(52)+acd66(51)+acd66(13)+acd66(50)
      acd66(50)=acd66(1)*acd66(50)
      acd66(51)=-acd66(8)*acd66(16)
      acd66(52)=-acd66(6)*acd66(36)
      acd66(53)=-acd66(12)*acd66(18)
      acd66(54)=acd66(10)*acd66(37)
      acd66(51)=acd66(54)+acd66(53)+acd66(51)+acd66(52)
      acd66(51)=acd66(25)*acd66(51)
      acd66(52)=-acd66(8)*acd66(26)
      acd66(53)=-acd66(6)*acd66(38)
      acd66(54)=-acd66(12)*acd66(27)
      acd66(55)=acd66(10)*acd66(39)
      acd66(52)=acd66(55)+acd66(54)+acd66(52)+acd66(53)
      acd66(52)=acd66(15)*acd66(52)
      acd66(53)=acd66(24)-acd66(34)
      acd66(53)=acd66(25)*acd66(53)
      acd66(54)=acd66(14)-acd66(30)
      acd66(54)=acd66(15)*acd66(54)
      acd66(53)=acd66(53)+acd66(54)
      acd66(53)=acd66(3)*acd66(53)
      acd66(50)=acd66(51)+acd66(52)+acd66(53)+acd66(50)
      acd66(51)=acd66(23)*acd66(29)
      acd66(52)=acd66(21)*acd66(28)
      acd66(53)=acd66(17)*acd66(26)
      acd66(51)=acd66(53)+acd66(51)-acd66(52)
      acd66(52)=acd66(27)*acd66(19)
      acd66(52)=acd66(52)+acd66(51)
      acd66(52)=acd66(24)*acd66(52)
      acd66(53)=acd66(22)*acd66(23)
      acd66(54)=acd66(21)*acd66(20)
      acd66(55)=acd66(17)*acd66(16)
      acd66(53)=acd66(55)+acd66(53)-acd66(54)
      acd66(54)=acd66(18)*acd66(19)
      acd66(54)=acd66(54)+acd66(53)
      acd66(54)=acd66(14)*acd66(54)
      acd66(55)=acd66(33)*acd66(35)
      acd66(56)=acd66(27)*acd66(31)
      acd66(51)=acd66(55)+acd66(56)-acd66(51)
      acd66(51)=acd66(34)*acd66(51)
      acd66(55)=acd66(33)*acd66(32)
      acd66(56)=acd66(18)*acd66(31)
      acd66(53)=acd66(56)+acd66(55)-acd66(53)
      acd66(53)=acd66(30)*acd66(53)
      acd66(55)=acd66(45)*acd66(49)
      acd66(56)=acd66(43)*acd66(47)
      acd66(55)=acd66(55)+acd66(56)
      acd66(55)=acd66(48)*acd66(55)
      acd66(56)=-acd66(27)*acd66(40)
      acd66(57)=-acd66(18)*acd66(42)
      acd66(56)=acd66(57)+acd66(56)
      acd66(56)=acd66(41)*acd66(56)
      acd66(57)=acd66(29)*acd66(43)
      acd66(58)=acd66(22)*acd66(45)
      acd66(57)=acd66(57)+acd66(58)
      acd66(57)=acd66(12)*acd66(57)
      acd66(58)=-acd66(29)*acd66(44)
      acd66(59)=-acd66(22)*acd66(46)
      acd66(58)=acd66(58)+acd66(59)
      acd66(58)=acd66(10)*acd66(58)
      brack=2.0_ki*acd66(50)+acd66(51)+acd66(52)+acd66(53)+acd66(54)+acd66(55)+&
      &acd66(56)+acd66(57)+acd66(58)
   end function brack_3
!---#] function brack_3:
!---#[ function brack_4:
   pure function brack_4(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd66h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(32) :: acd66
      complex(ki) :: brack
      acd66(1)=d(iv1,iv2)
      acd66(2)=k1(iv3)
      acd66(3)=abb66(38)
      acd66(4)=k2(iv3)
      acd66(5)=spval4k1(iv3)
      acd66(6)=abb66(30)
      acd66(7)=spval4k2(iv3)
      acd66(8)=abb66(12)
      acd66(9)=spval5k1(iv3)
      acd66(10)=abb66(28)
      acd66(11)=spval5k2(iv3)
      acd66(12)=abb66(21)
      acd66(13)=d(iv1,iv3)
      acd66(14)=k1(iv2)
      acd66(15)=k2(iv2)
      acd66(16)=spval4k1(iv2)
      acd66(17)=spval4k2(iv2)
      acd66(18)=spval5k1(iv2)
      acd66(19)=spval5k2(iv2)
      acd66(20)=d(iv2,iv3)
      acd66(21)=k1(iv1)
      acd66(22)=k2(iv1)
      acd66(23)=spval4k1(iv1)
      acd66(24)=spval4k2(iv1)
      acd66(25)=spval5k1(iv1)
      acd66(26)=spval5k2(iv1)
      acd66(27)=acd66(5)*acd66(6)
      acd66(28)=acd66(7)*acd66(8)
      acd66(29)=-acd66(9)*acd66(10)
      acd66(30)=acd66(11)*acd66(12)
      acd66(27)=acd66(30)+acd66(29)+acd66(28)+acd66(27)
      acd66(27)=acd66(1)*acd66(27)
      acd66(28)=acd66(16)*acd66(6)
      acd66(29)=acd66(17)*acd66(8)
      acd66(30)=-acd66(18)*acd66(10)
      acd66(31)=acd66(19)*acd66(12)
      acd66(28)=acd66(31)+acd66(30)+acd66(29)+acd66(28)
      acd66(28)=acd66(13)*acd66(28)
      acd66(29)=acd66(23)*acd66(6)
      acd66(30)=acd66(24)*acd66(8)
      acd66(31)=-acd66(25)*acd66(10)
      acd66(32)=acd66(26)*acd66(12)
      acd66(29)=acd66(32)+acd66(31)+acd66(30)+acd66(29)
      acd66(29)=acd66(20)*acd66(29)
      acd66(30)=acd66(22)-acd66(21)
      acd66(30)=acd66(30)*acd66(20)
      acd66(31)=acd66(15)-acd66(14)
      acd66(31)=acd66(31)*acd66(13)
      acd66(32)=-acd66(2)+acd66(4)
      acd66(32)=acd66(32)*acd66(1)
      acd66(30)=acd66(32)+acd66(31)+acd66(30)
      acd66(30)=acd66(3)*acd66(30)
      acd66(27)=acd66(29)+acd66(28)+acd66(27)+acd66(30)
      brack=2.0_ki*acd66(27)
   end function brack_4
!---#] function brack_4:
!---#[ function derivative:
   function derivative(mu2,i1,i2,i3) result(numerator)
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd66h0_qp
      implicit none
      complex(ki), intent(in) :: mu2
      integer, intent(in), optional :: i1
      integer, intent(in), optional :: i2
      integer, intent(in), optional :: i3
      complex(ki) :: numerator
      complex(ki) :: loc
      integer :: t1
      integer :: deg
      complex(ki), dimension(4), parameter :: Q = (/ (0.0_ki,0.0_ki),(0.0_ki,0.&
      &0_ki),(0.0_ki,0.0_ki),(0.0_ki,0.0_ki)/)
      qshift = k3+k5
      numerator = 0.0_ki
      deg = 0
      if(present(i1)) then
          iv1=i1
          deg=1
      else
          iv1=1
      end if
      if(present(i2)) then
          iv2=i2
          deg=2
      else
          iv2=1
      end if
      if(present(i3)) then
          iv3=i3
          deg=3
      else
          iv3=1
      end if
      t1 = 0
      if(deg.eq.0) then
         numerator = cond(epspow.eq.t1,brack_1,Q,mu2)
         return
      end if
      if(deg.eq.1) then
         numerator = cond(epspow.eq.t1,brack_2,Q,mu2)
         return
      end if
      if(deg.eq.2) then
         numerator = cond(epspow.eq.t1,brack_3,Q,mu2)
         return
      end if
      if(deg.eq.3) then
         numerator = cond(epspow.eq.t1,brack_4,Q,mu2)
         return
      end if
   end function derivative
!---#] function derivative:
end module     p2_gg_httbar_d66h0l1d_qp
