module     p2_gg_httbar_d71h12l1d_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity12d71h12l1d_qp.f90
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
      use p2_gg_httbar_abbrevd71h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(1) :: acd71
      complex(ki) :: brack
      acd71(1)=abb71(15)
      brack=acd71(1)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd71h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(114) :: acd71
      complex(ki) :: brack
      acd71(1)=k2(iv1)
      acd71(2)=abb71(50)
      acd71(3)=l5(iv1)
      acd71(4)=abb71(27)
      acd71(5)=e2(iv1)
      acd71(6)=abb71(25)
      acd71(7)=spvak1k2(iv1)
      acd71(8)=abb71(11)
      acd71(9)=spvak1l3(iv1)
      acd71(10)=abb71(39)
      acd71(11)=spvak1l5(iv1)
      acd71(12)=abb71(24)
      acd71(13)=spvak2k1(iv1)
      acd71(14)=abb71(20)
      acd71(15)=spvak2l3(iv1)
      acd71(16)=abb71(13)
      acd71(17)=spvak2l4(iv1)
      acd71(18)=abb71(18)
      acd71(19)=spvak2l5(iv1)
      acd71(20)=abb71(10)
      acd71(21)=spval3k1(iv1)
      acd71(22)=abb71(54)
      acd71(23)=spval3k2(iv1)
      acd71(24)=abb71(40)
      acd71(25)=spval3l4(iv1)
      acd71(26)=abb71(52)
      acd71(27)=spval3l5(iv1)
      acd71(28)=abb71(19)
      acd71(29)=spval4k2(iv1)
      acd71(30)=abb71(89)
      acd71(31)=spval4l3(iv1)
      acd71(32)=abb71(55)
      acd71(33)=spval4l5(iv1)
      acd71(34)=abb71(72)
      acd71(35)=spval5k1(iv1)
      acd71(36)=abb71(65)
      acd71(37)=spval5k2(iv1)
      acd71(38)=abb71(38)
      acd71(39)=spval5l4(iv1)
      acd71(40)=abb71(126)
      acd71(41)=spvak1e2(iv1)
      acd71(42)=abb71(14)
      acd71(43)=spvae2k1(iv1)
      acd71(44)=abb71(124)
      acd71(45)=spvak2e1(iv1)
      acd71(46)=abb71(122)
      acd71(47)=spvae1k2(iv1)
      acd71(48)=abb71(121)
      acd71(49)=spvak2e2(iv1)
      acd71(50)=abb71(34)
      acd71(51)=spvae2k2(iv1)
      acd71(52)=abb71(30)
      acd71(53)=spval3e1(iv1)
      acd71(54)=abb71(102)
      acd71(55)=spvae1l3(iv1)
      acd71(56)=abb71(66)
      acd71(57)=spval3e2(iv1)
      acd71(58)=abb71(78)
      acd71(59)=spvae2l3(iv1)
      acd71(60)=abb71(109)
      acd71(61)=spval4e2(iv1)
      acd71(62)=abb71(91)
      acd71(63)=spvae2l4(iv1)
      acd71(64)=abb71(85)
      acd71(65)=spval5e1(iv1)
      acd71(66)=abb71(56)
      acd71(67)=spvae1l5(iv1)
      acd71(68)=abb71(81)
      acd71(69)=spval5e2(iv1)
      acd71(70)=abb71(79)
      acd71(71)=spvae2l5(iv1)
      acd71(72)=abb71(42)
      acd71(73)=spvae1e2(iv1)
      acd71(74)=abb71(49)
      acd71(75)=spvae2e1(iv1)
      acd71(76)=abb71(43)
      acd71(77)=-acd71(2)*acd71(1)
      acd71(78)=-acd71(4)*acd71(3)
      acd71(79)=-acd71(6)*acd71(5)
      acd71(80)=-acd71(8)*acd71(7)
      acd71(81)=-acd71(10)*acd71(9)
      acd71(82)=-acd71(12)*acd71(11)
      acd71(83)=-acd71(14)*acd71(13)
      acd71(84)=-acd71(16)*acd71(15)
      acd71(85)=-acd71(18)*acd71(17)
      acd71(86)=-acd71(20)*acd71(19)
      acd71(87)=-acd71(22)*acd71(21)
      acd71(88)=-acd71(24)*acd71(23)
      acd71(89)=-acd71(26)*acd71(25)
      acd71(90)=-acd71(28)*acd71(27)
      acd71(91)=-acd71(30)*acd71(29)
      acd71(92)=-acd71(32)*acd71(31)
      acd71(93)=-acd71(34)*acd71(33)
      acd71(94)=-acd71(36)*acd71(35)
      acd71(95)=-acd71(38)*acd71(37)
      acd71(96)=-acd71(40)*acd71(39)
      acd71(97)=-acd71(42)*acd71(41)
      acd71(98)=-acd71(44)*acd71(43)
      acd71(99)=-acd71(46)*acd71(45)
      acd71(100)=-acd71(48)*acd71(47)
      acd71(101)=-acd71(50)*acd71(49)
      acd71(102)=-acd71(52)*acd71(51)
      acd71(103)=-acd71(54)*acd71(53)
      acd71(104)=-acd71(56)*acd71(55)
      acd71(105)=-acd71(58)*acd71(57)
      acd71(106)=-acd71(60)*acd71(59)
      acd71(107)=-acd71(62)*acd71(61)
      acd71(108)=-acd71(64)*acd71(63)
      acd71(109)=-acd71(66)*acd71(65)
      acd71(110)=-acd71(68)*acd71(67)
      acd71(111)=-acd71(70)*acd71(69)
      acd71(112)=-acd71(72)*acd71(71)
      acd71(113)=-acd71(74)*acd71(73)
      acd71(114)=-acd71(76)*acd71(75)
      brack=acd71(77)+acd71(78)+acd71(79)+acd71(80)+acd71(81)+acd71(82)+acd71(8&
      &3)+acd71(84)+acd71(85)+acd71(86)+acd71(87)+acd71(88)+acd71(89)+acd71(90)&
      &+acd71(91)+acd71(92)+acd71(93)+acd71(94)+acd71(95)+acd71(96)+acd71(97)+a&
      &cd71(98)+acd71(99)+acd71(100)+acd71(101)+acd71(102)+acd71(103)+acd71(104&
      &)+acd71(105)+acd71(106)+acd71(107)+acd71(108)+acd71(109)+acd71(110)+acd7&
      &1(111)+acd71(112)+acd71(113)+acd71(114)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd71h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(121) :: acd71
      complex(ki) :: brack
      acd71(1)=d(iv1,iv2)
      acd71(2)=abb71(29)
      acd71(3)=k2(iv1)
      acd71(4)=e2(iv2)
      acd71(5)=abb71(26)
      acd71(6)=spvak2e2(iv2)
      acd71(7)=abb71(9)
      acd71(8)=spvae2k2(iv2)
      acd71(9)=abb71(115)
      acd71(10)=k2(iv2)
      acd71(11)=e2(iv1)
      acd71(12)=spvak2e2(iv1)
      acd71(13)=spvae2k2(iv1)
      acd71(14)=l5(iv1)
      acd71(15)=abb71(103)
      acd71(16)=spvae2l5(iv2)
      acd71(17)=abb71(31)
      acd71(18)=l5(iv2)
      acd71(19)=spvae2l5(iv1)
      acd71(20)=spvak1l3(iv2)
      acd71(21)=abb71(37)
      acd71(22)=spvak1l5(iv2)
      acd71(23)=abb71(28)
      acd71(24)=spvak2k1(iv2)
      acd71(25)=abb71(23)
      acd71(26)=spvak2l3(iv2)
      acd71(27)=abb71(17)
      acd71(28)=spvak2l4(iv2)
      acd71(29)=abb71(33)
      acd71(30)=spvak2l5(iv2)
      acd71(31)=abb71(12)
      acd71(32)=spval3k1(iv2)
      acd71(33)=abb71(60)
      acd71(34)=spval3k2(iv2)
      acd71(35)=abb71(47)
      acd71(36)=spval3l4(iv2)
      acd71(37)=abb71(64)
      acd71(38)=spval3l5(iv2)
      acd71(39)=abb71(44)
      acd71(40)=spval4l3(iv2)
      acd71(41)=abb71(80)
      acd71(42)=spval4l5(iv2)
      acd71(43)=abb71(77)
      acd71(44)=spvak2e1(iv2)
      acd71(45)=abb71(67)
      acd71(46)=spval3e1(iv2)
      acd71(47)=abb71(112)
      acd71(48)=spvae1l3(iv2)
      acd71(49)=abb71(111)
      acd71(50)=spvae1l5(iv2)
      acd71(51)=abb71(82)
      acd71(52)=spvak1l3(iv1)
      acd71(53)=spvak1l5(iv1)
      acd71(54)=spvak2k1(iv1)
      acd71(55)=spvak2l3(iv1)
      acd71(56)=spvak2l4(iv1)
      acd71(57)=spvak2l5(iv1)
      acd71(58)=spval3k1(iv1)
      acd71(59)=spval3k2(iv1)
      acd71(60)=spval3l4(iv1)
      acd71(61)=spval3l5(iv1)
      acd71(62)=spval4l3(iv1)
      acd71(63)=spval4l5(iv1)
      acd71(64)=spvak2e1(iv1)
      acd71(65)=spval3e1(iv1)
      acd71(66)=spvae1l3(iv1)
      acd71(67)=spvae1l5(iv1)
      acd71(68)=spvak1k2(iv2)
      acd71(69)=abb71(22)
      acd71(70)=spval4k2(iv2)
      acd71(71)=abb71(62)
      acd71(72)=spvae1k2(iv2)
      acd71(73)=abb71(98)
      acd71(74)=spvak1k2(iv1)
      acd71(75)=spval4k2(iv1)
      acd71(76)=spvae1k2(iv1)
      acd71(77)=abb71(21)
      acd71(78)=abb71(53)
      acd71(79)=abb71(16)
      acd71(80)=abb71(51)
      acd71(81)=spval5k1(iv2)
      acd71(82)=abb71(58)
      acd71(83)=spval5k2(iv2)
      acd71(84)=abb71(113)
      acd71(85)=spval5l4(iv2)
      acd71(86)=abb71(59)
      acd71(87)=spval5e1(iv2)
      acd71(88)=abb71(68)
      acd71(89)=spval5k1(iv1)
      acd71(90)=spval5k2(iv1)
      acd71(91)=spval5l4(iv1)
      acd71(92)=spval5e1(iv1)
      acd71(93)=spval3e2(iv2)
      acd71(94)=abb71(125)
      acd71(95)=spval3e2(iv1)
      acd71(96)=spval5e2(iv2)
      acd71(97)=spval5e2(iv1)
      acd71(98)=abb71(119)
      acd71(99)=spvae2l3(iv2)
      acd71(100)=spvae2l3(iv1)
      acd71(101)=abb71(90)
      acd71(102)=abb71(75)
      acd71(103)=acd71(50)*acd71(51)
      acd71(104)=acd71(48)*acd71(49)
      acd71(105)=acd71(46)*acd71(47)
      acd71(106)=acd71(44)*acd71(45)
      acd71(107)=acd71(42)*acd71(43)
      acd71(108)=acd71(40)*acd71(41)
      acd71(109)=acd71(39)*acd71(38)
      acd71(110)=acd71(36)*acd71(37)
      acd71(111)=acd71(34)*acd71(35)
      acd71(112)=acd71(32)*acd71(33)
      acd71(113)=acd71(28)*acd71(29)
      acd71(114)=acd71(27)*acd71(26)
      acd71(115)=acd71(24)*acd71(25)
      acd71(116)=acd71(22)*acd71(23)
      acd71(117)=acd71(20)*acd71(21)
      acd71(118)=acd71(15)*acd71(18)
      acd71(119)=acd71(30)*acd71(31)
      acd71(120)=acd71(10)*acd71(5)
      acd71(103)=acd71(120)+acd71(119)+acd71(118)+acd71(117)+acd71(116)+acd71(1&
      &15)+acd71(114)+acd71(113)+acd71(112)+acd71(111)+acd71(110)+acd71(109)+ac&
      &d71(108)+acd71(107)+acd71(106)+acd71(105)+acd71(103)+acd71(104)
      acd71(103)=acd71(11)*acd71(103)
      acd71(104)=acd71(51)*acd71(67)
      acd71(105)=acd71(49)*acd71(66)
      acd71(106)=acd71(47)*acd71(65)
      acd71(107)=acd71(45)*acd71(64)
      acd71(108)=acd71(43)*acd71(63)
      acd71(109)=acd71(41)*acd71(62)
      acd71(110)=acd71(39)*acd71(61)
      acd71(111)=acd71(37)*acd71(60)
      acd71(112)=acd71(35)*acd71(59)
      acd71(113)=acd71(33)*acd71(58)
      acd71(114)=acd71(29)*acd71(56)
      acd71(115)=acd71(27)*acd71(55)
      acd71(116)=acd71(25)*acd71(54)
      acd71(117)=acd71(23)*acd71(53)
      acd71(118)=acd71(21)*acd71(52)
      acd71(119)=acd71(14)*acd71(15)
      acd71(120)=acd71(57)*acd71(31)
      acd71(121)=acd71(3)*acd71(5)
      acd71(104)=acd71(121)+acd71(120)+acd71(119)+acd71(118)+acd71(117)+acd71(1&
      &16)+acd71(115)+acd71(114)+acd71(113)+acd71(112)+acd71(111)+acd71(110)+ac&
      &d71(109)+acd71(108)+acd71(107)+acd71(106)+acd71(104)+acd71(105)
      acd71(104)=acd71(4)*acd71(104)
      acd71(105)=-acd71(17)*acd71(18)
      acd71(106)=-acd71(88)*acd71(87)
      acd71(107)=acd71(86)*acd71(85)
      acd71(108)=acd71(84)*acd71(83)
      acd71(109)=-acd71(82)*acd71(81)
      acd71(105)=acd71(109)+acd71(108)+acd71(107)+acd71(105)+acd71(106)
      acd71(105)=acd71(19)*acd71(105)
      acd71(106)=-acd71(14)*acd71(17)
      acd71(107)=-acd71(88)*acd71(92)
      acd71(108)=acd71(86)*acd71(91)
      acd71(109)=acd71(84)*acd71(90)
      acd71(110)=-acd71(82)*acd71(89)
      acd71(106)=acd71(110)+acd71(109)+acd71(108)+acd71(106)+acd71(107)
      acd71(106)=acd71(16)*acd71(106)
      acd71(107)=acd71(44)*acd71(80)
      acd71(108)=acd71(28)*acd71(78)
      acd71(109)=acd71(24)*acd71(77)
      acd71(110)=acd71(30)*acd71(79)
      acd71(111)=acd71(10)*acd71(9)
      acd71(107)=acd71(111)+acd71(110)+acd71(109)+acd71(107)+acd71(108)
      acd71(107)=acd71(13)*acd71(107)
      acd71(108)=acd71(64)*acd71(80)
      acd71(109)=acd71(56)*acd71(78)
      acd71(110)=acd71(54)*acd71(77)
      acd71(111)=acd71(57)*acd71(79)
      acd71(112)=acd71(3)*acd71(9)
      acd71(108)=acd71(112)+acd71(111)+acd71(110)+acd71(108)+acd71(109)
      acd71(108)=acd71(8)*acd71(108)
      acd71(109)=acd71(95)*acd71(20)
      acd71(110)=acd71(93)*acd71(52)
      acd71(111)=acd71(97)*acd71(22)
      acd71(112)=acd71(96)*acd71(53)
      acd71(109)=acd71(112)+acd71(111)+acd71(109)+acd71(110)
      acd71(109)=acd71(94)*acd71(109)
      acd71(110)=acd71(73)*acd71(72)
      acd71(111)=acd71(71)*acd71(70)
      acd71(112)=acd71(69)*acd71(68)
      acd71(113)=acd71(10)*acd71(7)
      acd71(110)=acd71(113)+acd71(112)+acd71(110)+acd71(111)
      acd71(110)=acd71(12)*acd71(110)
      acd71(111)=acd71(73)*acd71(76)
      acd71(112)=acd71(71)*acd71(75)
      acd71(113)=acd71(69)*acd71(74)
      acd71(114)=acd71(3)*acd71(7)
      acd71(111)=acd71(114)+acd71(113)+acd71(111)+acd71(112)
      acd71(111)=acd71(6)*acd71(111)
      acd71(112)=-acd71(30)*acd71(98)
      acd71(113)=-acd71(102)*acd71(50)
      acd71(114)=-acd71(101)*acd71(42)
      acd71(112)=acd71(114)+acd71(112)+acd71(113)
      acd71(112)=acd71(97)*acd71(112)
      acd71(113)=-acd71(57)*acd71(98)
      acd71(114)=-acd71(102)*acd71(67)
      acd71(115)=-acd71(101)*acd71(63)
      acd71(113)=acd71(115)+acd71(113)+acd71(114)
      acd71(113)=acd71(96)*acd71(113)
      acd71(114)=-acd71(95)*acd71(48)
      acd71(115)=-acd71(93)*acd71(66)
      acd71(114)=acd71(114)+acd71(115)
      acd71(114)=acd71(102)*acd71(114)
      acd71(115)=-acd71(95)*acd71(40)
      acd71(116)=-acd71(93)*acd71(62)
      acd71(115)=acd71(115)+acd71(116)
      acd71(115)=acd71(101)*acd71(115)
      acd71(116)=-acd71(100)*acd71(46)
      acd71(117)=-acd71(99)*acd71(65)
      acd71(116)=acd71(116)+acd71(117)
      acd71(116)=acd71(88)*acd71(116)
      acd71(117)=acd71(100)*acd71(36)
      acd71(118)=acd71(99)*acd71(60)
      acd71(117)=acd71(117)+acd71(118)
      acd71(117)=acd71(86)*acd71(117)
      acd71(118)=acd71(100)*acd71(34)
      acd71(119)=acd71(99)*acd71(59)
      acd71(118)=acd71(118)+acd71(119)
      acd71(118)=acd71(84)*acd71(118)
      acd71(119)=-acd71(100)*acd71(32)
      acd71(120)=-acd71(99)*acd71(58)
      acd71(119)=acd71(119)+acd71(120)
      acd71(119)=acd71(82)*acd71(119)
      acd71(120)=acd71(1)*acd71(2)
      brack=acd71(103)+acd71(104)+acd71(105)+acd71(106)+acd71(107)+acd71(108)+a&
      &cd71(109)+acd71(110)+acd71(111)+acd71(112)+acd71(113)+acd71(114)+acd71(1&
      &15)+acd71(116)+acd71(117)+acd71(118)+acd71(119)+2.0_ki*acd71(120)
   end function brack_3
!---#] function brack_3:
!---#[ function brack_4:
   pure function brack_4(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd71h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(55) :: acd71
      complex(ki) :: brack
      acd71(1)=d(iv1,iv2)
      acd71(2)=e2(iv3)
      acd71(3)=abb71(32)
      acd71(4)=spvak1e2(iv3)
      acd71(5)=abb71(125)
      acd71(6)=spvae2k1(iv3)
      acd71(7)=abb71(58)
      acd71(8)=spvak2e2(iv3)
      acd71(9)=abb71(119)
      acd71(10)=spvae2k2(iv3)
      acd71(11)=abb71(113)
      acd71(12)=spval4e2(iv3)
      acd71(13)=abb71(90)
      acd71(14)=spvae2l4(iv3)
      acd71(15)=abb71(59)
      acd71(16)=spvae2l5(iv3)
      acd71(17)=abb71(48)
      acd71(18)=spvae1e2(iv3)
      acd71(19)=abb71(75)
      acd71(20)=spvae2e1(iv3)
      acd71(21)=abb71(68)
      acd71(22)=d(iv1,iv3)
      acd71(23)=e2(iv2)
      acd71(24)=spvak1e2(iv2)
      acd71(25)=spvae2k1(iv2)
      acd71(26)=spvak2e2(iv2)
      acd71(27)=spvae2k2(iv2)
      acd71(28)=spval4e2(iv2)
      acd71(29)=spvae2l4(iv2)
      acd71(30)=spvae2l5(iv2)
      acd71(31)=spvae1e2(iv2)
      acd71(32)=spvae2e1(iv2)
      acd71(33)=d(iv2,iv3)
      acd71(34)=e2(iv1)
      acd71(35)=spvak1e2(iv1)
      acd71(36)=spvae2k1(iv1)
      acd71(37)=spvak2e2(iv1)
      acd71(38)=spvae2k2(iv1)
      acd71(39)=spval4e2(iv1)
      acd71(40)=spvae2l4(iv1)
      acd71(41)=spvae2l5(iv1)
      acd71(42)=spvae1e2(iv1)
      acd71(43)=spvae2e1(iv1)
      acd71(44)=-acd71(2)*acd71(3)
      acd71(45)=-acd71(4)*acd71(5)
      acd71(46)=acd71(6)*acd71(7)
      acd71(47)=acd71(8)*acd71(9)
      acd71(48)=-acd71(10)*acd71(11)
      acd71(49)=acd71(12)*acd71(13)
      acd71(50)=-acd71(14)*acd71(15)
      acd71(51)=-acd71(16)*acd71(17)
      acd71(52)=acd71(18)*acd71(19)
      acd71(53)=acd71(20)*acd71(21)
      acd71(44)=acd71(53)+acd71(52)+acd71(51)+acd71(50)+acd71(49)+acd71(48)+acd&
      &71(47)+acd71(46)+acd71(44)+acd71(45)
      acd71(44)=acd71(1)*acd71(44)
      acd71(45)=-acd71(23)*acd71(3)
      acd71(46)=-acd71(24)*acd71(5)
      acd71(47)=acd71(25)*acd71(7)
      acd71(48)=acd71(26)*acd71(9)
      acd71(49)=-acd71(27)*acd71(11)
      acd71(50)=acd71(28)*acd71(13)
      acd71(51)=-acd71(29)*acd71(15)
      acd71(52)=-acd71(30)*acd71(17)
      acd71(53)=acd71(31)*acd71(19)
      acd71(54)=acd71(32)*acd71(21)
      acd71(45)=acd71(54)+acd71(53)+acd71(52)+acd71(51)+acd71(50)+acd71(49)+acd&
      &71(48)+acd71(47)+acd71(46)+acd71(45)
      acd71(45)=acd71(22)*acd71(45)
      acd71(46)=-acd71(34)*acd71(3)
      acd71(47)=-acd71(35)*acd71(5)
      acd71(48)=acd71(36)*acd71(7)
      acd71(49)=acd71(37)*acd71(9)
      acd71(50)=-acd71(38)*acd71(11)
      acd71(51)=acd71(39)*acd71(13)
      acd71(52)=-acd71(40)*acd71(15)
      acd71(53)=-acd71(41)*acd71(17)
      acd71(54)=acd71(42)*acd71(19)
      acd71(55)=acd71(43)*acd71(21)
      acd71(46)=acd71(55)+acd71(54)+acd71(53)+acd71(52)+acd71(51)+acd71(50)+acd&
      &71(49)+acd71(48)+acd71(47)+acd71(46)
      acd71(46)=acd71(33)*acd71(46)
      acd71(44)=acd71(46)+acd71(45)+acd71(44)
      brack=2.0_ki*acd71(44)
   end function brack_4
!---#] function brack_4:
!---#[ function derivative:
   function derivative(mu2,i1,i2,i3) result(numerator)
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd71h12_qp
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
      qshift = 0
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
end module     p2_gg_httbar_d71h12l1d_qp
