module     p2_gg_httbar_d74h0l1d
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d74h0l1d.f90
   ! generator: buildfortran_d.py
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_util, only: cond, d => metric_tensor
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
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd74h0
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(1) :: acd74
      complex(ki) :: brack
      acd74(1)=abb74(15)
      brack=acd74(1)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd74h0
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(114) :: acd74
      complex(ki) :: brack
      acd74(1)=k2(iv1)
      acd74(2)=abb74(50)
      acd74(3)=l4(iv1)
      acd74(4)=abb74(27)
      acd74(5)=e2(iv1)
      acd74(6)=abb74(25)
      acd74(7)=spvak1k2(iv1)
      acd74(8)=abb74(52)
      acd74(9)=spvak1l3(iv1)
      acd74(10)=abb74(41)
      acd74(11)=spvak1l4(iv1)
      acd74(12)=abb74(36)
      acd74(13)=spvak2k1(iv1)
      acd74(14)=abb74(11)
      acd74(15)=spvak2l3(iv1)
      acd74(16)=abb74(17)
      acd74(17)=spvak2l4(iv1)
      acd74(18)=abb74(10)
      acd74(19)=spvak2l5(iv1)
      acd74(20)=abb74(39)
      acd74(21)=spval3k1(iv1)
      acd74(22)=abb74(24)
      acd74(23)=spval3k2(iv1)
      acd74(24)=abb74(13)
      acd74(25)=spval3l5(iv1)
      acd74(26)=abb74(38)
      acd74(27)=spval4k1(iv1)
      acd74(28)=abb74(28)
      acd74(29)=spval4k2(iv1)
      acd74(30)=abb74(12)
      acd74(31)=spval4l3(iv1)
      acd74(32)=abb74(19)
      acd74(33)=spval4l5(iv1)
      acd74(34)=abb74(107)
      acd74(35)=spval5k2(iv1)
      acd74(36)=abb74(128)
      acd74(37)=spval5l3(iv1)
      acd74(38)=abb74(64)
      acd74(39)=spval5l4(iv1)
      acd74(40)=abb74(126)
      acd74(41)=spvak1e2(iv1)
      acd74(42)=abb74(14)
      acd74(43)=spvae2k1(iv1)
      acd74(44)=abb74(125)
      acd74(45)=spvak2e1(iv1)
      acd74(46)=abb74(123)
      acd74(47)=spvae1k2(iv1)
      acd74(48)=abb74(121)
      acd74(49)=spvak2e2(iv1)
      acd74(50)=abb74(30)
      acd74(51)=spvae2k2(iv1)
      acd74(52)=abb74(43)
      acd74(53)=spval3e1(iv1)
      acd74(54)=abb74(66)
      acd74(55)=spvae1l3(iv1)
      acd74(56)=abb74(102)
      acd74(57)=spval3e2(iv1)
      acd74(58)=abb74(109)
      acd74(59)=spvae2l3(iv1)
      acd74(60)=abb74(81)
      acd74(61)=spval4e1(iv1)
      acd74(62)=abb74(90)
      acd74(63)=spvae1l4(iv1)
      acd74(64)=abb74(56)
      acd74(65)=spval4e2(iv1)
      acd74(66)=abb74(34)
      acd74(67)=spvae2l4(iv1)
      acd74(68)=abb74(82)
      acd74(69)=spval5e2(iv1)
      acd74(70)=abb74(78)
      acd74(71)=spvae2l5(iv1)
      acd74(72)=abb74(77)
      acd74(73)=spvae1e2(iv1)
      acd74(74)=abb74(42)
      acd74(75)=spvae2e1(iv1)
      acd74(76)=abb74(49)
      acd74(77)=-acd74(2)*acd74(1)
      acd74(78)=-acd74(4)*acd74(3)
      acd74(79)=-acd74(6)*acd74(5)
      acd74(80)=-acd74(8)*acd74(7)
      acd74(81)=-acd74(10)*acd74(9)
      acd74(82)=-acd74(12)*acd74(11)
      acd74(83)=-acd74(14)*acd74(13)
      acd74(84)=-acd74(16)*acd74(15)
      acd74(85)=-acd74(18)*acd74(17)
      acd74(86)=-acd74(20)*acd74(19)
      acd74(87)=-acd74(22)*acd74(21)
      acd74(88)=-acd74(24)*acd74(23)
      acd74(89)=-acd74(26)*acd74(25)
      acd74(90)=-acd74(28)*acd74(27)
      acd74(91)=-acd74(30)*acd74(29)
      acd74(92)=-acd74(32)*acd74(31)
      acd74(93)=-acd74(34)*acd74(33)
      acd74(94)=-acd74(36)*acd74(35)
      acd74(95)=-acd74(38)*acd74(37)
      acd74(96)=-acd74(40)*acd74(39)
      acd74(97)=-acd74(42)*acd74(41)
      acd74(98)=-acd74(44)*acd74(43)
      acd74(99)=-acd74(46)*acd74(45)
      acd74(100)=-acd74(48)*acd74(47)
      acd74(101)=-acd74(50)*acd74(49)
      acd74(102)=-acd74(52)*acd74(51)
      acd74(103)=-acd74(54)*acd74(53)
      acd74(104)=-acd74(56)*acd74(55)
      acd74(105)=-acd74(58)*acd74(57)
      acd74(106)=-acd74(60)*acd74(59)
      acd74(107)=-acd74(62)*acd74(61)
      acd74(108)=-acd74(64)*acd74(63)
      acd74(109)=-acd74(66)*acd74(65)
      acd74(110)=-acd74(68)*acd74(67)
      acd74(111)=-acd74(70)*acd74(69)
      acd74(112)=-acd74(72)*acd74(71)
      acd74(113)=-acd74(74)*acd74(73)
      acd74(114)=-acd74(76)*acd74(75)
      brack=acd74(77)+acd74(78)+acd74(79)+acd74(80)+acd74(81)+acd74(82)+acd74(8&
      &3)+acd74(84)+acd74(85)+acd74(86)+acd74(87)+acd74(88)+acd74(89)+acd74(90)&
      &+acd74(91)+acd74(92)+acd74(93)+acd74(94)+acd74(95)+acd74(96)+acd74(97)+a&
      &cd74(98)+acd74(99)+acd74(100)+acd74(101)+acd74(102)+acd74(103)+acd74(104&
      &)+acd74(105)+acd74(106)+acd74(107)+acd74(108)+acd74(109)+acd74(110)+acd7&
      &4(111)+acd74(112)+acd74(113)+acd74(114)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd74h0
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(121) :: acd74
      complex(ki) :: brack
      acd74(1)=d(iv1,iv2)
      acd74(2)=abb74(29)
      acd74(3)=k2(iv1)
      acd74(4)=e2(iv2)
      acd74(5)=abb74(26)
      acd74(6)=spvak2e2(iv2)
      acd74(7)=abb74(115)
      acd74(8)=spvae2k2(iv2)
      acd74(9)=abb74(9)
      acd74(10)=k2(iv2)
      acd74(11)=e2(iv1)
      acd74(12)=spvak2e2(iv1)
      acd74(13)=spvae2k2(iv1)
      acd74(14)=l4(iv1)
      acd74(15)=abb74(103)
      acd74(16)=spval4e2(iv2)
      acd74(17)=abb74(31)
      acd74(18)=l4(iv2)
      acd74(19)=spval4e2(iv1)
      acd74(20)=spvak1k2(iv2)
      acd74(21)=abb74(23)
      acd74(22)=spvak1l3(iv2)
      acd74(23)=abb74(47)
      acd74(24)=spvak2l3(iv2)
      acd74(25)=abb74(40)
      acd74(26)=spval3k1(iv2)
      acd74(27)=abb74(37)
      acd74(28)=spval3k2(iv2)
      acd74(29)=abb74(20)
      acd74(30)=spval3l5(iv2)
      acd74(31)=abb74(46)
      acd74(32)=spval4k1(iv2)
      acd74(33)=abb74(33)
      acd74(34)=spval4k2(iv2)
      acd74(35)=abb74(16)
      acd74(36)=spval4l3(iv2)
      acd74(37)=abb74(44)
      acd74(38)=spval4l5(iv2)
      acd74(39)=abb74(130)
      acd74(40)=spval5k2(iv2)
      acd74(41)=abb74(129)
      acd74(42)=spval5l3(iv2)
      acd74(43)=abb74(127)
      acd74(44)=spvae1k2(iv2)
      acd74(45)=abb74(67)
      acd74(46)=spval3e1(iv2)
      acd74(47)=abb74(112)
      acd74(48)=spvae1l3(iv2)
      acd74(49)=abb74(110)
      acd74(50)=spval4e1(iv2)
      acd74(51)=abb74(91)
      acd74(52)=spvak1k2(iv1)
      acd74(53)=spvak1l3(iv1)
      acd74(54)=spvak2l3(iv1)
      acd74(55)=spval3k1(iv1)
      acd74(56)=spval3k2(iv1)
      acd74(57)=spval3l5(iv1)
      acd74(58)=spval4k1(iv1)
      acd74(59)=spval4k2(iv1)
      acd74(60)=spval4l3(iv1)
      acd74(61)=spval4l5(iv1)
      acd74(62)=spval5k2(iv1)
      acd74(63)=spval5l3(iv1)
      acd74(64)=spvae1k2(iv1)
      acd74(65)=spval3e1(iv1)
      acd74(66)=spvae1l3(iv1)
      acd74(67)=spval4e1(iv1)
      acd74(68)=abb74(21)
      acd74(69)=abb74(18)
      acd74(70)=abb74(63)
      acd74(71)=abb74(51)
      acd74(72)=spvak2k1(iv2)
      acd74(73)=abb74(22)
      acd74(74)=spvak2l5(iv2)
      acd74(75)=abb74(45)
      acd74(76)=spvak2e1(iv2)
      acd74(77)=abb74(98)
      acd74(78)=spvak2k1(iv1)
      acd74(79)=spvak2l5(iv1)
      acd74(80)=spvak2e1(iv1)
      acd74(81)=spvak1l4(iv2)
      acd74(82)=abb74(58)
      acd74(83)=spvak2l4(iv2)
      acd74(84)=abb74(120)
      acd74(85)=spval5l4(iv2)
      acd74(86)=abb74(59)
      acd74(87)=spvae1l4(iv2)
      acd74(88)=abb74(68)
      acd74(89)=spvak1l4(iv1)
      acd74(90)=spvak2l4(iv1)
      acd74(91)=spval5l4(iv1)
      acd74(92)=spvae1l4(iv1)
      acd74(93)=spval3e2(iv2)
      acd74(94)=spval3e2(iv1)
      acd74(95)=spvae2l3(iv2)
      acd74(96)=abb74(124)
      acd74(97)=spvae2l3(iv1)
      acd74(98)=abb74(61)
      acd74(99)=spvae2l4(iv2)
      acd74(100)=spvae2l4(iv1)
      acd74(101)=abb74(118)
      acd74(102)=abb74(75)
      acd74(103)=acd74(50)*acd74(51)
      acd74(104)=acd74(48)*acd74(49)
      acd74(105)=acd74(46)*acd74(47)
      acd74(106)=acd74(44)*acd74(45)
      acd74(107)=acd74(42)*acd74(43)
      acd74(108)=acd74(40)*acd74(41)
      acd74(109)=acd74(38)*acd74(39)
      acd74(110)=acd74(37)*acd74(36)
      acd74(111)=acd74(32)*acd74(33)
      acd74(112)=acd74(30)*acd74(31)
      acd74(113)=acd74(29)*acd74(28)
      acd74(114)=acd74(26)*acd74(27)
      acd74(115)=acd74(24)*acd74(25)
      acd74(116)=acd74(22)*acd74(23)
      acd74(117)=acd74(20)*acd74(21)
      acd74(118)=acd74(15)*acd74(18)
      acd74(119)=acd74(34)*acd74(35)
      acd74(120)=acd74(10)*acd74(5)
      acd74(103)=acd74(120)+acd74(119)+acd74(118)+acd74(117)+acd74(116)+acd74(1&
      &15)+acd74(114)+acd74(113)+acd74(112)+acd74(111)+acd74(110)+acd74(109)+ac&
      &d74(108)+acd74(107)+acd74(106)+acd74(105)+acd74(103)+acd74(104)
      acd74(103)=acd74(11)*acd74(103)
      acd74(104)=acd74(51)*acd74(67)
      acd74(105)=acd74(49)*acd74(66)
      acd74(106)=acd74(47)*acd74(65)
      acd74(107)=acd74(45)*acd74(64)
      acd74(108)=acd74(43)*acd74(63)
      acd74(109)=acd74(41)*acd74(62)
      acd74(110)=acd74(39)*acd74(61)
      acd74(111)=acd74(37)*acd74(60)
      acd74(112)=acd74(33)*acd74(58)
      acd74(113)=acd74(31)*acd74(57)
      acd74(114)=acd74(29)*acd74(56)
      acd74(115)=acd74(27)*acd74(55)
      acd74(116)=acd74(25)*acd74(54)
      acd74(117)=acd74(23)*acd74(53)
      acd74(118)=acd74(21)*acd74(52)
      acd74(119)=acd74(14)*acd74(15)
      acd74(120)=acd74(59)*acd74(35)
      acd74(121)=acd74(3)*acd74(5)
      acd74(104)=acd74(121)+acd74(120)+acd74(119)+acd74(118)+acd74(117)+acd74(1&
      &16)+acd74(115)+acd74(114)+acd74(113)+acd74(112)+acd74(111)+acd74(110)+ac&
      &d74(109)+acd74(108)+acd74(107)+acd74(106)+acd74(104)+acd74(105)
      acd74(104)=acd74(4)*acd74(104)
      acd74(105)=-acd74(17)*acd74(18)
      acd74(106)=-acd74(88)*acd74(87)
      acd74(107)=acd74(86)*acd74(85)
      acd74(108)=acd74(84)*acd74(83)
      acd74(109)=-acd74(82)*acd74(81)
      acd74(105)=acd74(109)+acd74(108)+acd74(107)+acd74(105)+acd74(106)
      acd74(105)=acd74(19)*acd74(105)
      acd74(106)=-acd74(14)*acd74(17)
      acd74(107)=-acd74(88)*acd74(92)
      acd74(108)=acd74(86)*acd74(91)
      acd74(109)=acd74(84)*acd74(90)
      acd74(110)=-acd74(82)*acd74(89)
      acd74(106)=acd74(110)+acd74(109)+acd74(108)+acd74(106)+acd74(107)
      acd74(106)=acd74(16)*acd74(106)
      acd74(107)=acd74(44)*acd74(71)
      acd74(108)=acd74(40)*acd74(70)
      acd74(109)=acd74(20)*acd74(68)
      acd74(110)=acd74(34)*acd74(69)
      acd74(111)=acd74(10)*acd74(7)
      acd74(107)=acd74(111)+acd74(110)+acd74(109)+acd74(107)+acd74(108)
      acd74(107)=acd74(12)*acd74(107)
      acd74(108)=acd74(64)*acd74(71)
      acd74(109)=acd74(62)*acd74(70)
      acd74(110)=acd74(52)*acd74(68)
      acd74(111)=acd74(59)*acd74(69)
      acd74(112)=acd74(3)*acd74(7)
      acd74(108)=acd74(112)+acd74(111)+acd74(110)+acd74(108)+acd74(109)
      acd74(108)=acd74(6)*acd74(108)
      acd74(109)=-acd74(97)*acd74(30)
      acd74(110)=-acd74(95)*acd74(57)
      acd74(111)=-acd74(100)*acd74(38)
      acd74(112)=-acd74(99)*acd74(61)
      acd74(109)=acd74(112)+acd74(111)+acd74(109)+acd74(110)
      acd74(109)=acd74(98)*acd74(109)
      acd74(110)=acd74(97)*acd74(26)
      acd74(111)=acd74(95)*acd74(55)
      acd74(112)=acd74(100)*acd74(32)
      acd74(113)=acd74(99)*acd74(58)
      acd74(110)=acd74(113)+acd74(112)+acd74(110)+acd74(111)
      acd74(110)=acd74(96)*acd74(110)
      acd74(111)=acd74(77)*acd74(76)
      acd74(112)=acd74(75)*acd74(74)
      acd74(113)=acd74(73)*acd74(72)
      acd74(114)=acd74(10)*acd74(9)
      acd74(111)=acd74(114)+acd74(113)+acd74(111)+acd74(112)
      acd74(111)=acd74(13)*acd74(111)
      acd74(112)=acd74(77)*acd74(80)
      acd74(113)=acd74(75)*acd74(79)
      acd74(114)=acd74(73)*acd74(78)
      acd74(115)=acd74(3)*acd74(9)
      acd74(112)=acd74(115)+acd74(114)+acd74(112)+acd74(113)
      acd74(112)=acd74(8)*acd74(112)
      acd74(113)=-acd74(97)*acd74(46)
      acd74(114)=-acd74(95)*acd74(65)
      acd74(113)=acd74(113)+acd74(114)
      acd74(113)=acd74(102)*acd74(113)
      acd74(114)=-acd74(34)*acd74(101)
      acd74(115)=-acd74(102)*acd74(50)
      acd74(114)=acd74(114)+acd74(115)
      acd74(114)=acd74(100)*acd74(114)
      acd74(115)=-acd74(59)*acd74(101)
      acd74(116)=-acd74(102)*acd74(67)
      acd74(115)=acd74(115)+acd74(116)
      acd74(115)=acd74(99)*acd74(115)
      acd74(116)=-acd74(94)*acd74(48)
      acd74(117)=-acd74(93)*acd74(66)
      acd74(116)=acd74(116)+acd74(117)
      acd74(116)=acd74(88)*acd74(116)
      acd74(117)=acd74(94)*acd74(42)
      acd74(118)=acd74(93)*acd74(63)
      acd74(117)=acd74(117)+acd74(118)
      acd74(117)=acd74(86)*acd74(117)
      acd74(118)=acd74(94)*acd74(24)
      acd74(119)=acd74(93)*acd74(54)
      acd74(118)=acd74(118)+acd74(119)
      acd74(118)=acd74(84)*acd74(118)
      acd74(119)=-acd74(94)*acd74(22)
      acd74(120)=-acd74(93)*acd74(53)
      acd74(119)=acd74(119)+acd74(120)
      acd74(119)=acd74(82)*acd74(119)
      acd74(120)=acd74(1)*acd74(2)
      brack=acd74(103)+acd74(104)+acd74(105)+acd74(106)+acd74(107)+acd74(108)+a&
      &cd74(109)+acd74(110)+acd74(111)+acd74(112)+acd74(113)+acd74(114)+acd74(1&
      &15)+acd74(116)+acd74(117)+acd74(118)+acd74(119)+2.0_ki*acd74(120)
   end function brack_3
!---#] function brack_3:
!---#[ function brack_4:
   pure function brack_4(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd74h0
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(55) :: acd74
      complex(ki) :: brack
      acd74(1)=d(iv1,iv2)
      acd74(2)=e2(iv3)
      acd74(3)=abb74(32)
      acd74(4)=spvak1e2(iv3)
      acd74(5)=abb74(58)
      acd74(6)=spvae2k1(iv3)
      acd74(7)=abb74(124)
      acd74(8)=spvak2e2(iv3)
      acd74(9)=abb74(120)
      acd74(10)=spvae2k2(iv3)
      acd74(11)=abb74(118)
      acd74(12)=spval4e2(iv3)
      acd74(13)=abb74(48)
      acd74(14)=spval5e2(iv3)
      acd74(15)=abb74(59)
      acd74(16)=spvae2l5(iv3)
      acd74(17)=abb74(61)
      acd74(18)=spvae1e2(iv3)
      acd74(19)=abb74(68)
      acd74(20)=spvae2e1(iv3)
      acd74(21)=abb74(75)
      acd74(22)=d(iv1,iv3)
      acd74(23)=e2(iv2)
      acd74(24)=spvak1e2(iv2)
      acd74(25)=spvae2k1(iv2)
      acd74(26)=spvak2e2(iv2)
      acd74(27)=spvae2k2(iv2)
      acd74(28)=spval4e2(iv2)
      acd74(29)=spval5e2(iv2)
      acd74(30)=spvae2l5(iv2)
      acd74(31)=spvae1e2(iv2)
      acd74(32)=spvae2e1(iv2)
      acd74(33)=d(iv2,iv3)
      acd74(34)=e2(iv1)
      acd74(35)=spvak1e2(iv1)
      acd74(36)=spvae2k1(iv1)
      acd74(37)=spvak2e2(iv1)
      acd74(38)=spvae2k2(iv1)
      acd74(39)=spval4e2(iv1)
      acd74(40)=spval5e2(iv1)
      acd74(41)=spvae2l5(iv1)
      acd74(42)=spvae1e2(iv1)
      acd74(43)=spvae2e1(iv1)
      acd74(44)=-acd74(2)*acd74(3)
      acd74(45)=acd74(4)*acd74(5)
      acd74(46)=-acd74(6)*acd74(7)
      acd74(47)=-acd74(8)*acd74(9)
      acd74(48)=acd74(10)*acd74(11)
      acd74(49)=-acd74(12)*acd74(13)
      acd74(50)=-acd74(14)*acd74(15)
      acd74(51)=acd74(16)*acd74(17)
      acd74(52)=acd74(18)*acd74(19)
      acd74(53)=acd74(20)*acd74(21)
      acd74(44)=acd74(53)+acd74(52)+acd74(51)+acd74(50)+acd74(49)+acd74(48)+acd&
      &74(47)+acd74(46)+acd74(44)+acd74(45)
      acd74(44)=acd74(1)*acd74(44)
      acd74(45)=-acd74(23)*acd74(3)
      acd74(46)=acd74(24)*acd74(5)
      acd74(47)=-acd74(25)*acd74(7)
      acd74(48)=-acd74(26)*acd74(9)
      acd74(49)=acd74(27)*acd74(11)
      acd74(50)=-acd74(28)*acd74(13)
      acd74(51)=-acd74(29)*acd74(15)
      acd74(52)=acd74(30)*acd74(17)
      acd74(53)=acd74(31)*acd74(19)
      acd74(54)=acd74(32)*acd74(21)
      acd74(45)=acd74(54)+acd74(53)+acd74(52)+acd74(51)+acd74(50)+acd74(49)+acd&
      &74(48)+acd74(47)+acd74(46)+acd74(45)
      acd74(45)=acd74(22)*acd74(45)
      acd74(46)=-acd74(34)*acd74(3)
      acd74(47)=acd74(35)*acd74(5)
      acd74(48)=-acd74(36)*acd74(7)
      acd74(49)=-acd74(37)*acd74(9)
      acd74(50)=acd74(38)*acd74(11)
      acd74(51)=-acd74(39)*acd74(13)
      acd74(52)=-acd74(40)*acd74(15)
      acd74(53)=acd74(41)*acd74(17)
      acd74(54)=acd74(42)*acd74(19)
      acd74(55)=acd74(43)*acd74(21)
      acd74(46)=acd74(55)+acd74(54)+acd74(53)+acd74(52)+acd74(51)+acd74(50)+acd&
      &74(49)+acd74(48)+acd74(47)+acd74(46)
      acd74(46)=acd74(33)*acd74(46)
      acd74(44)=acd74(46)+acd74(45)+acd74(44)
      brack=2.0_ki*acd74(44)
   end function brack_4
!---#] function brack_4:
!---#[ function derivative:
   function derivative(mu2,i1,i2,i3) result(numerator)
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd74h0
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
end module     p2_gg_httbar_d74h0l1d
