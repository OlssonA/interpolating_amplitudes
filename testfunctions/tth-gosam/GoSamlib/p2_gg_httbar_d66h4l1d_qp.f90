module     p2_gg_httbar_d66h4l1d_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d66h4l1d_qp.f90
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
      use p2_gg_httbar_abbrevd66h4_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(80) :: acd66
      complex(ki) :: brack
      acd66(1)=dotproduct(k1,qshift)
      acd66(2)=dotproduct(k2,qshift)
      acd66(3)=abb66(35)
      acd66(4)=dotproduct(qshift,qshift)
      acd66(5)=abb66(30)
      acd66(6)=dotproduct(qshift,spvak2l3)
      acd66(7)=abb66(22)
      acd66(8)=dotproduct(qshift,spval3k2)
      acd66(9)=abb66(41)
      acd66(10)=dotproduct(qshift,spval3l4)
      acd66(11)=abb66(56)
      acd66(12)=dotproduct(qshift,spval5l3)
      acd66(13)=abb66(23)
      acd66(14)=dotproduct(qshift,spval5l4)
      acd66(15)=abb66(33)
      acd66(16)=abb66(51)
      acd66(17)=abb66(20)
      acd66(18)=abb66(19)
      acd66(19)=dotproduct(qshift,spvak1k2)
      acd66(20)=abb66(24)
      acd66(21)=abb66(9)
      acd66(22)=dotproduct(l3,qshift)
      acd66(23)=abb66(48)
      acd66(24)=abb66(38)
      acd66(25)=dotproduct(qshift,spvak1l4)
      acd66(26)=abb66(42)
      acd66(27)=dotproduct(qshift,spvak2k1)
      acd66(28)=abb66(31)
      acd66(29)=dotproduct(qshift,spval5k1)
      acd66(30)=abb66(58)
      acd66(31)=abb66(18)
      acd66(32)=abb66(26)
      acd66(33)=abb66(10)
      acd66(34)=dotproduct(qshift,spvak1l3)
      acd66(35)=abb66(11)
      acd66(36)=abb66(17)
      acd66(37)=dotproduct(qshift,spval3k1)
      acd66(38)=abb66(34)
      acd66(39)=abb66(32)
      acd66(40)=abb66(14)
      acd66(41)=abb66(40)
      acd66(42)=dotproduct(qshift,spval5k2)
      acd66(43)=abb66(29)
      acd66(44)=abb66(28)
      acd66(45)=abb66(50)
      acd66(46)=abb66(21)
      acd66(47)=dotproduct(qshift,spvak1l5)
      acd66(48)=abb66(39)
      acd66(49)=abb66(43)
      acd66(50)=dotproduct(qshift,spvak2l4)
      acd66(51)=abb66(13)
      acd66(52)=dotproduct(qshift,spvak2l5)
      acd66(53)=abb66(15)
      acd66(54)=abb66(25)
      acd66(55)=dotproduct(qshift,spval3l5)
      acd66(56)=abb66(16)
      acd66(57)=dotproduct(qshift,spval4k1)
      acd66(58)=abb66(52)
      acd66(59)=abb66(12)
      acd66(60)=acd66(14)*acd66(15)
      acd66(61)=acd66(10)*acd66(11)
      acd66(62)=acd66(12)*acd66(13)
      acd66(63)=acd66(6)*acd66(7)
      acd66(64)=acd66(8)*acd66(9)
      acd66(60)=-acd66(64)+acd66(60)-acd66(61)+acd66(62)+acd66(63)
      acd66(61)=-acd66(4)*acd66(5)
      acd66(61)=acd66(61)-acd66(16)+acd66(60)
      acd66(61)=acd66(1)*acd66(61)
      acd66(62)=acd66(19)*acd66(20)
      acd66(63)=-acd66(4)*acd66(18)
      acd66(64)=acd66(1)*acd66(3)
      acd66(65)=acd66(2)*acd66(17)
      acd66(60)=acd66(65)+acd66(64)+acd66(63)-acd66(21)+acd66(62)-acd66(60)
      acd66(60)=acd66(2)*acd66(60)
      acd66(62)=-acd66(29)*acd66(30)
      acd66(63)=acd66(25)*acd66(26)
      acd66(64)=-acd66(27)*acd66(28)
      acd66(65)=-acd66(19)*acd66(24)
      acd66(62)=acd66(65)+acd66(64)+acd66(63)+acd66(31)+acd66(62)
      acd66(62)=acd66(4)*acd66(62)
      acd66(63)=acd66(24)*acd66(34)
      acd66(64)=acd66(6)*acd66(32)
      acd66(63)=acd66(64)-acd66(35)+acd66(63)
      acd66(63)=acd66(8)*acd66(63)
      acd66(64)=-acd66(57)*acd66(58)
      acd66(65)=-acd66(55)*acd66(56)
      acd66(66)=-acd66(52)*acd66(53)
      acd66(67)=-acd66(50)*acd66(51)
      acd66(68)=-acd66(47)*acd66(48)
      acd66(69)=-acd66(22)*acd66(23)
      acd66(70)=-acd66(42)*acd66(49)
      acd66(71)=-acd66(37)*acd66(54)
      acd66(72)=-acd66(34)*acd66(46)
      acd66(73)=-acd66(29)*acd66(45)
      acd66(74)=-acd66(25)*acd66(41)
      acd66(75)=acd66(42)*acd66(43)
      acd66(75)=-acd66(44)+acd66(75)
      acd66(75)=acd66(27)*acd66(75)
      acd66(76)=-acd66(19)*acd66(40)
      acd66(77)=-acd66(14)*acd66(39)
      acd66(78)=-acd66(10)*acd66(36)
      acd66(79)=acd66(30)*acd66(37)
      acd66(79)=-acd66(38)+acd66(79)
      acd66(79)=acd66(12)*acd66(79)
      acd66(80)=-acd66(6)*acd66(33)
      brack=acd66(59)+acd66(60)+acd66(61)+acd66(62)+acd66(63)+acd66(64)+acd66(6&
      &5)+acd66(66)+acd66(67)+acd66(68)+acd66(69)+acd66(70)+acd66(71)+acd66(72)&
      &+acd66(73)+acd66(74)+acd66(75)+acd66(76)+acd66(77)+acd66(78)+acd66(79)+a&
      &cd66(80)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd66h4_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(99) :: acd66
      complex(ki) :: brack
      acd66(1)=k1(iv1)
      acd66(2)=dotproduct(k2,qshift)
      acd66(3)=abb66(35)
      acd66(4)=dotproduct(qshift,qshift)
      acd66(5)=abb66(30)
      acd66(6)=dotproduct(qshift,spvak2l3)
      acd66(7)=abb66(22)
      acd66(8)=dotproduct(qshift,spval3k2)
      acd66(9)=abb66(41)
      acd66(10)=dotproduct(qshift,spval3l4)
      acd66(11)=abb66(56)
      acd66(12)=dotproduct(qshift,spval5l3)
      acd66(13)=abb66(23)
      acd66(14)=dotproduct(qshift,spval5l4)
      acd66(15)=abb66(33)
      acd66(16)=abb66(51)
      acd66(17)=k2(iv1)
      acd66(18)=dotproduct(k1,qshift)
      acd66(19)=abb66(20)
      acd66(20)=abb66(19)
      acd66(21)=dotproduct(qshift,spvak1k2)
      acd66(22)=abb66(24)
      acd66(23)=abb66(9)
      acd66(24)=l3(iv1)
      acd66(25)=abb66(48)
      acd66(26)=qshift(iv1)
      acd66(27)=abb66(38)
      acd66(28)=dotproduct(qshift,spvak1l4)
      acd66(29)=abb66(42)
      acd66(30)=dotproduct(qshift,spvak2k1)
      acd66(31)=abb66(31)
      acd66(32)=dotproduct(qshift,spval5k1)
      acd66(33)=abb66(58)
      acd66(34)=abb66(18)
      acd66(35)=spvak2l3(iv1)
      acd66(36)=abb66(26)
      acd66(37)=abb66(10)
      acd66(38)=spval3k2(iv1)
      acd66(39)=dotproduct(qshift,spvak1l3)
      acd66(40)=abb66(11)
      acd66(41)=spval3l4(iv1)
      acd66(42)=abb66(17)
      acd66(43)=spval5l3(iv1)
      acd66(44)=dotproduct(qshift,spval3k1)
      acd66(45)=abb66(34)
      acd66(46)=spval5l4(iv1)
      acd66(47)=abb66(32)
      acd66(48)=spvak1k2(iv1)
      acd66(49)=abb66(14)
      acd66(50)=spvak1l4(iv1)
      acd66(51)=abb66(40)
      acd66(52)=spvak2k1(iv1)
      acd66(53)=dotproduct(qshift,spval5k2)
      acd66(54)=abb66(29)
      acd66(55)=abb66(28)
      acd66(56)=spval5k1(iv1)
      acd66(57)=abb66(50)
      acd66(58)=spvak1l3(iv1)
      acd66(59)=abb66(21)
      acd66(60)=spvak1l5(iv1)
      acd66(61)=abb66(39)
      acd66(62)=spval5k2(iv1)
      acd66(63)=abb66(43)
      acd66(64)=spvak2l4(iv1)
      acd66(65)=abb66(13)
      acd66(66)=spvak2l5(iv1)
      acd66(67)=abb66(15)
      acd66(68)=spval3k1(iv1)
      acd66(69)=abb66(25)
      acd66(70)=spval3l5(iv1)
      acd66(71)=abb66(16)
      acd66(72)=spval4k1(iv1)
      acd66(73)=abb66(52)
      acd66(74)=acd66(15)*acd66(46)
      acd66(75)=acd66(13)*acd66(43)
      acd66(76)=acd66(11)*acd66(41)
      acd66(77)=acd66(7)*acd66(35)
      acd66(78)=acd66(38)*acd66(9)
      acd66(74)=acd66(78)-acd66(74)-acd66(75)+acd66(76)-acd66(77)
      acd66(75)=2.0_ki*acd66(26)
      acd66(76)=acd66(5)*acd66(75)
      acd66(76)=acd66(76)+acd66(74)
      acd66(76)=acd66(18)*acd66(76)
      acd66(77)=acd66(15)*acd66(14)
      acd66(78)=acd66(13)*acd66(12)
      acd66(79)=acd66(11)*acd66(10)
      acd66(80)=acd66(8)*acd66(9)
      acd66(81)=acd66(7)*acd66(6)
      acd66(77)=-acd66(77)-acd66(78)+acd66(79)+acd66(80)-acd66(81)
      acd66(78)=acd66(4)*acd66(5)
      acd66(78)=acd66(78)+acd66(16)+acd66(77)
      acd66(78)=acd66(1)*acd66(78)
      acd66(79)=-acd66(48)*acd66(22)
      acd66(80)=acd66(20)*acd66(75)
      acd66(81)=-acd66(1)*acd66(3)
      acd66(74)=acd66(81)+acd66(80)+acd66(79)-acd66(74)
      acd66(74)=acd66(2)*acd66(74)
      acd66(79)=-acd66(21)*acd66(22)
      acd66(80)=acd66(4)*acd66(20)
      acd66(81)=-acd66(18)*acd66(3)
      acd66(82)=acd66(2)*acd66(19)
      acd66(77)=-2.0_ki*acd66(82)+acd66(81)+acd66(80)+acd66(23)+acd66(79)-acd66&
      &(77)
      acd66(77)=acd66(17)*acd66(77)
      acd66(79)=-acd66(29)*acd66(50)
      acd66(80)=acd66(52)*acd66(31)
      acd66(81)=acd66(33)*acd66(56)
      acd66(82)=acd66(27)*acd66(48)
      acd66(79)=acd66(82)+acd66(81)+acd66(79)+acd66(80)
      acd66(79)=acd66(4)*acd66(79)
      acd66(80)=acd66(30)*acd66(31)
      acd66(81)=-acd66(29)*acd66(28)
      acd66(82)=acd66(33)*acd66(32)
      acd66(83)=acd66(27)*acd66(21)
      acd66(80)=acd66(83)+acd66(82)+acd66(81)-acd66(34)+acd66(80)
      acd66(75)=acd66(80)*acd66(75)
      acd66(80)=-acd66(12)*acd66(68)
      acd66(81)=-acd66(43)*acd66(44)
      acd66(80)=acd66(80)+acd66(81)
      acd66(80)=acd66(33)*acd66(80)
      acd66(81)=-acd66(35)*acd66(36)
      acd66(82)=-acd66(27)*acd66(58)
      acd66(81)=acd66(81)+acd66(82)
      acd66(81)=acd66(8)*acd66(81)
      acd66(82)=-acd66(6)*acd66(36)
      acd66(83)=-acd66(27)*acd66(39)
      acd66(82)=acd66(83)+acd66(40)+acd66(82)
      acd66(82)=acd66(38)*acd66(82)
      acd66(83)=-acd66(30)*acd66(54)
      acd66(83)=acd66(83)+acd66(63)
      acd66(83)=acd66(62)*acd66(83)
      acd66(84)=acd66(72)*acd66(73)
      acd66(85)=acd66(70)*acd66(71)
      acd66(86)=acd66(66)*acd66(67)
      acd66(87)=acd66(64)*acd66(65)
      acd66(88)=acd66(60)*acd66(61)
      acd66(89)=acd66(24)*acd66(25)
      acd66(90)=acd66(68)*acd66(69)
      acd66(91)=acd66(58)*acd66(59)
      acd66(92)=acd66(56)*acd66(57)
      acd66(93)=acd66(50)*acd66(51)
      acd66(94)=-acd66(54)*acd66(53)
      acd66(94)=acd66(55)+acd66(94)
      acd66(94)=acd66(52)*acd66(94)
      acd66(95)=acd66(48)*acd66(49)
      acd66(96)=acd66(46)*acd66(47)
      acd66(97)=acd66(41)*acd66(42)
      acd66(98)=acd66(43)*acd66(45)
      acd66(99)=acd66(35)*acd66(37)
      brack=acd66(74)+acd66(75)+acd66(76)+acd66(77)+acd66(78)+acd66(79)+acd66(8&
      &0)+acd66(81)+acd66(82)+acd66(83)+acd66(84)+acd66(85)+acd66(86)+acd66(87)&
      &+acd66(88)+acd66(89)+acd66(90)+acd66(91)+acd66(92)+acd66(93)+acd66(94)+a&
      &cd66(95)+acd66(96)+acd66(97)+acd66(98)+acd66(99)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd66h4_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(66) :: acd66
      complex(ki) :: brack
      acd66(1)=d(iv1,iv2)
      acd66(2)=dotproduct(k1,qshift)
      acd66(3)=abb66(30)
      acd66(4)=dotproduct(k2,qshift)
      acd66(5)=abb66(19)
      acd66(6)=dotproduct(qshift,spvak1k2)
      acd66(7)=abb66(38)
      acd66(8)=dotproduct(qshift,spvak1l4)
      acd66(9)=abb66(42)
      acd66(10)=dotproduct(qshift,spvak2k1)
      acd66(11)=abb66(31)
      acd66(12)=dotproduct(qshift,spval5k1)
      acd66(13)=abb66(58)
      acd66(14)=abb66(18)
      acd66(15)=k1(iv1)
      acd66(16)=k2(iv2)
      acd66(17)=abb66(35)
      acd66(18)=qshift(iv2)
      acd66(19)=spvak2l3(iv2)
      acd66(20)=abb66(22)
      acd66(21)=spval3k2(iv2)
      acd66(22)=abb66(41)
      acd66(23)=spval3l4(iv2)
      acd66(24)=abb66(56)
      acd66(25)=spval5l3(iv2)
      acd66(26)=abb66(23)
      acd66(27)=spval5l4(iv2)
      acd66(28)=abb66(33)
      acd66(29)=k1(iv2)
      acd66(30)=k2(iv1)
      acd66(31)=qshift(iv1)
      acd66(32)=spvak2l3(iv1)
      acd66(33)=spval3k2(iv1)
      acd66(34)=spval3l4(iv1)
      acd66(35)=spval5l3(iv1)
      acd66(36)=spval5l4(iv1)
      acd66(37)=abb66(20)
      acd66(38)=spvak1k2(iv2)
      acd66(39)=abb66(24)
      acd66(40)=spvak1k2(iv1)
      acd66(41)=spvak1l4(iv2)
      acd66(42)=spvak2k1(iv2)
      acd66(43)=spval5k1(iv2)
      acd66(44)=spvak1l4(iv1)
      acd66(45)=spvak2k1(iv1)
      acd66(46)=spval5k1(iv1)
      acd66(47)=spval5k2(iv2)
      acd66(48)=abb66(29)
      acd66(49)=spval5k2(iv1)
      acd66(50)=abb66(26)
      acd66(51)=spvak1l3(iv2)
      acd66(52)=spvak1l3(iv1)
      acd66(53)=spval3k1(iv2)
      acd66(54)=spval3k1(iv1)
      acd66(55)=-acd66(11)*acd66(10)
      acd66(56)=acd66(9)*acd66(8)
      acd66(57)=-acd66(5)*acd66(4)
      acd66(58)=-acd66(3)*acd66(2)
      acd66(59)=-acd66(13)*acd66(12)
      acd66(60)=-acd66(7)*acd66(6)
      acd66(55)=acd66(60)+acd66(59)+acd66(58)+acd66(57)+acd66(56)+acd66(14)+acd&
      &66(55)
      acd66(55)=acd66(1)*acd66(55)
      acd66(56)=acd66(28)*acd66(36)
      acd66(57)=acd66(26)*acd66(35)
      acd66(58)=acd66(24)*acd66(34)
      acd66(59)=acd66(22)*acd66(33)
      acd66(60)=acd66(20)*acd66(32)
      acd66(56)=-acd66(56)-acd66(57)+acd66(58)+acd66(59)-acd66(60)
      acd66(57)=2.0_ki*acd66(31)
      acd66(58)=-acd66(3)*acd66(57)
      acd66(58)=acd66(58)-acd66(56)
      acd66(58)=acd66(29)*acd66(58)
      acd66(59)=acd66(28)*acd66(27)
      acd66(60)=acd66(26)*acd66(25)
      acd66(61)=acd66(24)*acd66(23)
      acd66(62)=acd66(21)*acd66(22)
      acd66(63)=acd66(20)*acd66(19)
      acd66(59)=-acd66(59)-acd66(60)+acd66(61)+acd66(62)-acd66(63)
      acd66(60)=2.0_ki*acd66(18)
      acd66(61)=-acd66(3)*acd66(60)
      acd66(61)=acd66(61)-acd66(59)
      acd66(61)=acd66(15)*acd66(61)
      acd66(62)=acd66(38)*acd66(39)
      acd66(63)=-acd66(5)*acd66(60)
      acd66(64)=acd66(29)*acd66(17)
      acd66(59)=acd66(64)+acd66(63)+acd66(62)+acd66(59)
      acd66(59)=acd66(30)*acd66(59)
      acd66(62)=acd66(39)*acd66(40)
      acd66(63)=-acd66(5)*acd66(57)
      acd66(64)=acd66(15)*acd66(17)
      acd66(65)=acd66(30)*acd66(37)
      acd66(56)=2.0_ki*acd66(65)+acd66(64)+acd66(63)+acd66(62)+acd66(56)
      acd66(56)=acd66(16)*acd66(56)
      acd66(62)=-acd66(11)*acd66(42)
      acd66(63)=acd66(9)*acd66(41)
      acd66(64)=-acd66(13)*acd66(43)
      acd66(65)=-acd66(7)*acd66(38)
      acd66(62)=acd66(65)+acd66(64)+acd66(62)+acd66(63)
      acd66(57)=acd66(62)*acd66(57)
      acd66(62)=-acd66(11)*acd66(45)
      acd66(63)=acd66(9)*acd66(44)
      acd66(64)=-acd66(13)*acd66(46)
      acd66(65)=-acd66(7)*acd66(40)
      acd66(62)=acd66(65)+acd66(64)+acd66(62)+acd66(63)
      acd66(60)=acd66(62)*acd66(60)
      acd66(62)=acd66(33)*acd66(19)
      acd66(63)=acd66(21)*acd66(32)
      acd66(62)=acd66(63)+acd66(62)
      acd66(62)=acd66(50)*acd66(62)
      acd66(63)=acd66(45)*acd66(47)
      acd66(64)=acd66(42)*acd66(49)
      acd66(63)=acd66(63)+acd66(64)
      acd66(63)=acd66(48)*acd66(63)
      acd66(64)=acd66(35)*acd66(53)
      acd66(65)=acd66(25)*acd66(54)
      acd66(64)=acd66(64)+acd66(65)
      acd66(64)=acd66(13)*acd66(64)
      acd66(65)=acd66(33)*acd66(51)
      acd66(66)=acd66(21)*acd66(52)
      acd66(65)=acd66(65)+acd66(66)
      acd66(65)=acd66(7)*acd66(65)
      brack=2.0_ki*acd66(55)+acd66(56)+acd66(57)+acd66(58)+acd66(59)+acd66(60)+&
      &acd66(61)+acd66(62)+acd66(63)+acd66(64)+acd66(65)
   end function brack_3
!---#] function brack_3:
!---#[ function brack_4:
   pure function brack_4(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd66h4_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(35) :: acd66
      complex(ki) :: brack
      acd66(1)=d(iv1,iv2)
      acd66(2)=k1(iv3)
      acd66(3)=abb66(30)
      acd66(4)=k2(iv3)
      acd66(5)=abb66(19)
      acd66(6)=spvak1k2(iv3)
      acd66(7)=abb66(38)
      acd66(8)=spvak1l4(iv3)
      acd66(9)=abb66(42)
      acd66(10)=spvak2k1(iv3)
      acd66(11)=abb66(31)
      acd66(12)=spval5k1(iv3)
      acd66(13)=abb66(58)
      acd66(14)=d(iv1,iv3)
      acd66(15)=k1(iv2)
      acd66(16)=k2(iv2)
      acd66(17)=spvak1k2(iv2)
      acd66(18)=spvak1l4(iv2)
      acd66(19)=spvak2k1(iv2)
      acd66(20)=spval5k1(iv2)
      acd66(21)=d(iv2,iv3)
      acd66(22)=k1(iv1)
      acd66(23)=k2(iv1)
      acd66(24)=spvak1k2(iv1)
      acd66(25)=spvak1l4(iv1)
      acd66(26)=spvak2k1(iv1)
      acd66(27)=spval5k1(iv1)
      acd66(28)=acd66(2)*acd66(3)
      acd66(29)=acd66(4)*acd66(5)
      acd66(30)=acd66(6)*acd66(7)
      acd66(31)=-acd66(8)*acd66(9)
      acd66(32)=acd66(10)*acd66(11)
      acd66(33)=acd66(12)*acd66(13)
      acd66(28)=acd66(33)+acd66(32)+acd66(31)+acd66(30)+acd66(28)+acd66(29)
      acd66(28)=acd66(1)*acd66(28)
      acd66(29)=acd66(15)*acd66(3)
      acd66(30)=acd66(16)*acd66(5)
      acd66(31)=acd66(17)*acd66(7)
      acd66(32)=-acd66(18)*acd66(9)
      acd66(33)=acd66(19)*acd66(11)
      acd66(34)=acd66(20)*acd66(13)
      acd66(29)=acd66(34)+acd66(33)+acd66(32)+acd66(31)+acd66(30)+acd66(29)
      acd66(29)=acd66(14)*acd66(29)
      acd66(30)=acd66(22)*acd66(3)
      acd66(31)=acd66(23)*acd66(5)
      acd66(32)=acd66(24)*acd66(7)
      acd66(33)=-acd66(25)*acd66(9)
      acd66(34)=acd66(26)*acd66(11)
      acd66(35)=acd66(27)*acd66(13)
      acd66(30)=acd66(35)+acd66(34)+acd66(33)+acd66(32)+acd66(31)+acd66(30)
      acd66(30)=acd66(21)*acd66(30)
      acd66(28)=acd66(30)+acd66(29)+acd66(28)
      brack=2.0_ki*acd66(28)
   end function brack_4
!---#] function brack_4:
!---#[ function derivative:
   function derivative(mu2,i1,i2,i3) result(numerator)
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd66h4_qp
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
end module     p2_gg_httbar_d66h4l1d_qp
