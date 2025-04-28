module     p2_gg_httbar_d31h4l1d
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d31h4l1d.f90
   ! generator: buildfortran_d.py
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_util, only: cond, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, private :: iv0
   integer, private :: iv1
   integer, private :: iv2
   real(ki), dimension(4), private :: qshift
   public :: derivative
contains
!---#[ function brack_1:
   pure function brack_1(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd31h4
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(129) :: acd31
      complex(ki) :: brack
      acd31(1)=dotproduct(k1,qshift)
      acd31(2)=abb31(49)
      acd31(3)=dotproduct(k2,qshift)
      acd31(4)=abb31(10)
      acd31(5)=dotproduct(l5,qshift)
      acd31(6)=abb31(16)
      acd31(7)=dotproduct(e2,qshift)
      acd31(8)=dotproduct(qshift,spvak1l3)
      acd31(9)=abb31(40)
      acd31(10)=dotproduct(qshift,spvak1l4)
      acd31(11)=abb31(27)
      acd31(12)=dotproduct(qshift,spvak2k1)
      acd31(13)=abb31(20)
      acd31(14)=dotproduct(qshift,spvak2l3)
      acd31(15)=abb31(31)
      acd31(16)=dotproduct(qshift,spvak2l4)
      acd31(17)=abb31(43)
      acd31(18)=dotproduct(qshift,spvak2l5)
      acd31(19)=abb31(30)
      acd31(20)=dotproduct(qshift,spval3k1)
      acd31(21)=abb31(67)
      acd31(22)=dotproduct(qshift,spval3l5)
      acd31(23)=abb31(74)
      acd31(24)=dotproduct(qshift,spval5l3)
      acd31(25)=abb31(48)
      acd31(26)=dotproduct(qshift,spval5l4)
      acd31(27)=abb31(56)
      acd31(28)=dotproduct(qshift,spvak2e1)
      acd31(29)=abb31(29)
      acd31(30)=dotproduct(qshift,spval3e1)
      acd31(31)=abb31(39)
      acd31(32)=dotproduct(qshift,spvae1l3)
      acd31(33)=abb31(44)
      acd31(34)=dotproduct(qshift,spvae1l4)
      acd31(35)=abb31(68)
      acd31(36)=abb31(18)
      acd31(37)=dotproduct(qshift,qshift)
      acd31(38)=abb31(28)
      acd31(39)=abb31(24)
      acd31(40)=abb31(25)
      acd31(41)=abb31(9)
      acd31(42)=abb31(19)
      acd31(43)=abb31(11)
      acd31(44)=abb31(23)
      acd31(45)=abb31(36)
      acd31(46)=abb31(73)
      acd31(47)=abb31(37)
      acd31(48)=abb31(46)
      acd31(49)=abb31(26)
      acd31(50)=dotproduct(qshift,spvak1k2)
      acd31(51)=abb31(50)
      acd31(52)=dotproduct(qshift,spvak1l5)
      acd31(53)=abb31(13)
      acd31(54)=dotproduct(qshift,spval3k2)
      acd31(55)=abb31(33)
      acd31(56)=dotproduct(qshift,spval5k1)
      acd31(57)=abb31(71)
      acd31(58)=dotproduct(qshift,spval5k2)
      acd31(59)=abb31(70)
      acd31(60)=dotproduct(qshift,spvak1e1)
      acd31(61)=abb31(32)
      acd31(62)=dotproduct(qshift,spvae1k1)
      acd31(63)=abb31(45)
      acd31(64)=dotproduct(qshift,spvak1e2)
      acd31(65)=abb31(42)
      acd31(66)=dotproduct(qshift,spvae2k1)
      acd31(67)=abb31(34)
      acd31(68)=dotproduct(qshift,spvae1k2)
      acd31(69)=abb31(15)
      acd31(70)=dotproduct(qshift,spvak2e2)
      acd31(71)=abb31(14)
      acd31(72)=dotproduct(qshift,spvae2k2)
      acd31(73)=abb31(58)
      acd31(74)=dotproduct(qshift,spval3e2)
      acd31(75)=abb31(72)
      acd31(76)=dotproduct(qshift,spvae2l3)
      acd31(77)=abb31(41)
      acd31(78)=dotproduct(qshift,spvae2l4)
      acd31(79)=abb31(22)
      acd31(80)=dotproduct(qshift,spval5e1)
      acd31(81)=abb31(63)
      acd31(82)=dotproduct(qshift,spvae1l5)
      acd31(83)=abb31(60)
      acd31(84)=dotproduct(qshift,spval5e2)
      acd31(85)=abb31(47)
      acd31(86)=dotproduct(qshift,spvae2l5)
      acd31(87)=abb31(35)
      acd31(88)=dotproduct(qshift,spvae1e2)
      acd31(89)=abb31(21)
      acd31(90)=dotproduct(qshift,spvae2e1)
      acd31(91)=abb31(12)
      acd31(92)=abb31(17)
      acd31(93)=acd31(9)*acd31(8)
      acd31(94)=acd31(11)*acd31(10)
      acd31(95)=acd31(13)*acd31(12)
      acd31(96)=acd31(15)*acd31(14)
      acd31(97)=acd31(17)*acd31(16)
      acd31(98)=acd31(19)*acd31(18)
      acd31(99)=acd31(21)*acd31(20)
      acd31(100)=acd31(23)*acd31(22)
      acd31(101)=acd31(25)*acd31(24)
      acd31(102)=acd31(27)*acd31(26)
      acd31(103)=acd31(29)*acd31(28)
      acd31(104)=acd31(31)*acd31(30)
      acd31(105)=acd31(33)*acd31(32)
      acd31(106)=acd31(35)*acd31(34)
      acd31(93)=-acd31(36)+acd31(106)+acd31(105)+acd31(104)+acd31(103)+acd31(10&
      &2)+acd31(101)+acd31(100)+acd31(99)+acd31(98)+acd31(97)+acd31(96)+acd31(9&
      &5)+acd31(94)+acd31(93)
      acd31(93)=acd31(7)*acd31(93)
      acd31(94)=acd31(2)*acd31(1)
      acd31(95)=-acd31(4)*acd31(3)
      acd31(96)=-acd31(6)*acd31(5)
      acd31(97)=acd31(38)*acd31(37)
      acd31(98)=-acd31(39)*acd31(8)
      acd31(99)=-acd31(40)*acd31(10)
      acd31(100)=-acd31(41)*acd31(12)
      acd31(101)=-acd31(42)*acd31(14)
      acd31(102)=-acd31(43)*acd31(16)
      acd31(103)=-acd31(44)*acd31(18)
      acd31(104)=-acd31(45)*acd31(20)
      acd31(105)=-acd31(46)*acd31(22)
      acd31(106)=-acd31(47)*acd31(24)
      acd31(107)=-acd31(48)*acd31(26)
      acd31(108)=-acd31(49)*acd31(28)
      acd31(109)=-acd31(51)*acd31(50)
      acd31(110)=-acd31(53)*acd31(52)
      acd31(111)=-acd31(55)*acd31(54)
      acd31(112)=-acd31(57)*acd31(56)
      acd31(113)=-acd31(59)*acd31(58)
      acd31(114)=-acd31(61)*acd31(60)
      acd31(115)=-acd31(63)*acd31(62)
      acd31(116)=-acd31(65)*acd31(64)
      acd31(117)=-acd31(67)*acd31(66)
      acd31(118)=-acd31(69)*acd31(68)
      acd31(119)=-acd31(71)*acd31(70)
      acd31(120)=-acd31(73)*acd31(72)
      acd31(121)=-acd31(75)*acd31(74)
      acd31(122)=-acd31(77)*acd31(76)
      acd31(123)=-acd31(79)*acd31(78)
      acd31(124)=-acd31(81)*acd31(80)
      acd31(125)=-acd31(83)*acd31(82)
      acd31(126)=-acd31(85)*acd31(84)
      acd31(127)=-acd31(87)*acd31(86)
      acd31(128)=-acd31(89)*acd31(88)
      acd31(129)=-acd31(91)*acd31(90)
      brack=acd31(92)+acd31(93)+acd31(94)+acd31(95)+acd31(96)+acd31(97)+acd31(9&
      &8)+acd31(99)+acd31(100)+acd31(101)+acd31(102)+acd31(103)+acd31(104)+acd3&
      &1(105)+acd31(106)+acd31(107)+acd31(108)+acd31(109)+acd31(110)+acd31(111)&
      &+acd31(112)+acd31(113)+acd31(114)+acd31(115)+acd31(116)+acd31(117)+acd31&
      &(118)+acd31(119)+acd31(120)+acd31(121)+acd31(122)+acd31(123)+acd31(124)+&
      &acd31(125)+acd31(126)+acd31(127)+acd31(128)+acd31(129)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd31h4
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(144) :: acd31
      complex(ki) :: brack
      acd31(1)=k1(iv1)
      acd31(2)=abb31(49)
      acd31(3)=k2(iv1)
      acd31(4)=abb31(10)
      acd31(5)=l5(iv1)
      acd31(6)=abb31(16)
      acd31(7)=e2(iv1)
      acd31(8)=dotproduct(qshift,spvak1l3)
      acd31(9)=abb31(40)
      acd31(10)=dotproduct(qshift,spvak1l4)
      acd31(11)=abb31(27)
      acd31(12)=dotproduct(qshift,spvak2k1)
      acd31(13)=abb31(20)
      acd31(14)=dotproduct(qshift,spvak2l3)
      acd31(15)=abb31(31)
      acd31(16)=dotproduct(qshift,spvak2l4)
      acd31(17)=abb31(43)
      acd31(18)=dotproduct(qshift,spvak2l5)
      acd31(19)=abb31(30)
      acd31(20)=dotproduct(qshift,spval3k1)
      acd31(21)=abb31(67)
      acd31(22)=dotproduct(qshift,spval3l5)
      acd31(23)=abb31(74)
      acd31(24)=dotproduct(qshift,spval5l3)
      acd31(25)=abb31(48)
      acd31(26)=dotproduct(qshift,spval5l4)
      acd31(27)=abb31(56)
      acd31(28)=dotproduct(qshift,spvak2e1)
      acd31(29)=abb31(29)
      acd31(30)=dotproduct(qshift,spval3e1)
      acd31(31)=abb31(39)
      acd31(32)=dotproduct(qshift,spvae1l3)
      acd31(33)=abb31(44)
      acd31(34)=dotproduct(qshift,spvae1l4)
      acd31(35)=abb31(68)
      acd31(36)=abb31(18)
      acd31(37)=qshift(iv1)
      acd31(38)=abb31(28)
      acd31(39)=spvak1l3(iv1)
      acd31(40)=dotproduct(e2,qshift)
      acd31(41)=abb31(24)
      acd31(42)=spvak1l4(iv1)
      acd31(43)=abb31(25)
      acd31(44)=spvak2k1(iv1)
      acd31(45)=abb31(9)
      acd31(46)=spvak2l3(iv1)
      acd31(47)=abb31(19)
      acd31(48)=spvak2l4(iv1)
      acd31(49)=abb31(11)
      acd31(50)=spvak2l5(iv1)
      acd31(51)=abb31(23)
      acd31(52)=spval3k1(iv1)
      acd31(53)=abb31(36)
      acd31(54)=spval3l5(iv1)
      acd31(55)=abb31(73)
      acd31(56)=spval5l3(iv1)
      acd31(57)=abb31(37)
      acd31(58)=spval5l4(iv1)
      acd31(59)=abb31(46)
      acd31(60)=spvak2e1(iv1)
      acd31(61)=abb31(26)
      acd31(62)=spval3e1(iv1)
      acd31(63)=spvae1l3(iv1)
      acd31(64)=spvae1l4(iv1)
      acd31(65)=spvak1k2(iv1)
      acd31(66)=abb31(50)
      acd31(67)=spvak1l5(iv1)
      acd31(68)=abb31(13)
      acd31(69)=spval3k2(iv1)
      acd31(70)=abb31(33)
      acd31(71)=spval5k1(iv1)
      acd31(72)=abb31(71)
      acd31(73)=spval5k2(iv1)
      acd31(74)=abb31(70)
      acd31(75)=spvak1e1(iv1)
      acd31(76)=abb31(32)
      acd31(77)=spvae1k1(iv1)
      acd31(78)=abb31(45)
      acd31(79)=spvak1e2(iv1)
      acd31(80)=abb31(42)
      acd31(81)=spvae2k1(iv1)
      acd31(82)=abb31(34)
      acd31(83)=spvae1k2(iv1)
      acd31(84)=abb31(15)
      acd31(85)=spvak2e2(iv1)
      acd31(86)=abb31(14)
      acd31(87)=spvae2k2(iv1)
      acd31(88)=abb31(58)
      acd31(89)=spval3e2(iv1)
      acd31(90)=abb31(72)
      acd31(91)=spvae2l3(iv1)
      acd31(92)=abb31(41)
      acd31(93)=spvae2l4(iv1)
      acd31(94)=abb31(22)
      acd31(95)=spval5e1(iv1)
      acd31(96)=abb31(63)
      acd31(97)=spvae1l5(iv1)
      acd31(98)=abb31(60)
      acd31(99)=spval5e2(iv1)
      acd31(100)=abb31(47)
      acd31(101)=spvae2l5(iv1)
      acd31(102)=abb31(35)
      acd31(103)=spvae1e2(iv1)
      acd31(104)=abb31(21)
      acd31(105)=spvae2e1(iv1)
      acd31(106)=abb31(12)
      acd31(107)=acd31(39)*acd31(9)
      acd31(108)=acd31(42)*acd31(11)
      acd31(109)=acd31(44)*acd31(13)
      acd31(110)=acd31(46)*acd31(15)
      acd31(111)=acd31(48)*acd31(17)
      acd31(112)=acd31(50)*acd31(19)
      acd31(113)=acd31(52)*acd31(21)
      acd31(114)=acd31(54)*acd31(23)
      acd31(115)=acd31(56)*acd31(25)
      acd31(116)=acd31(58)*acd31(27)
      acd31(117)=acd31(60)*acd31(29)
      acd31(118)=acd31(62)*acd31(31)
      acd31(119)=acd31(63)*acd31(33)
      acd31(120)=acd31(64)*acd31(35)
      acd31(107)=acd31(120)+acd31(119)+acd31(118)+acd31(117)+acd31(116)+acd31(1&
      &15)+acd31(114)+acd31(113)+acd31(112)+acd31(111)+acd31(110)+acd31(109)+ac&
      &d31(107)+acd31(108)
      acd31(107)=acd31(40)*acd31(107)
      acd31(108)=acd31(8)*acd31(9)
      acd31(109)=acd31(10)*acd31(11)
      acd31(110)=acd31(12)*acd31(13)
      acd31(111)=acd31(14)*acd31(15)
      acd31(112)=acd31(16)*acd31(17)
      acd31(113)=acd31(18)*acd31(19)
      acd31(114)=acd31(20)*acd31(21)
      acd31(115)=acd31(22)*acd31(23)
      acd31(116)=acd31(24)*acd31(25)
      acd31(117)=acd31(26)*acd31(27)
      acd31(118)=acd31(28)*acd31(29)
      acd31(119)=acd31(30)*acd31(31)
      acd31(120)=acd31(32)*acd31(33)
      acd31(121)=acd31(34)*acd31(35)
      acd31(108)=-acd31(36)+acd31(121)+acd31(120)+acd31(119)+acd31(118)+acd31(1&
      &17)+acd31(116)+acd31(115)+acd31(114)+acd31(113)+acd31(112)+acd31(111)+ac&
      &d31(110)+acd31(109)+acd31(108)
      acd31(108)=acd31(7)*acd31(108)
      acd31(109)=acd31(2)*acd31(1)
      acd31(110)=-acd31(4)*acd31(3)
      acd31(111)=-acd31(6)*acd31(5)
      acd31(112)=acd31(38)*acd31(37)
      acd31(113)=-acd31(41)*acd31(39)
      acd31(114)=-acd31(43)*acd31(42)
      acd31(115)=-acd31(45)*acd31(44)
      acd31(116)=-acd31(47)*acd31(46)
      acd31(117)=-acd31(49)*acd31(48)
      acd31(118)=-acd31(51)*acd31(50)
      acd31(119)=-acd31(53)*acd31(52)
      acd31(120)=-acd31(55)*acd31(54)
      acd31(121)=-acd31(57)*acd31(56)
      acd31(122)=-acd31(59)*acd31(58)
      acd31(123)=-acd31(61)*acd31(60)
      acd31(124)=-acd31(66)*acd31(65)
      acd31(125)=-acd31(68)*acd31(67)
      acd31(126)=-acd31(70)*acd31(69)
      acd31(127)=-acd31(72)*acd31(71)
      acd31(128)=-acd31(74)*acd31(73)
      acd31(129)=-acd31(76)*acd31(75)
      acd31(130)=-acd31(78)*acd31(77)
      acd31(131)=-acd31(80)*acd31(79)
      acd31(132)=-acd31(82)*acd31(81)
      acd31(133)=-acd31(84)*acd31(83)
      acd31(134)=-acd31(86)*acd31(85)
      acd31(135)=-acd31(88)*acd31(87)
      acd31(136)=-acd31(90)*acd31(89)
      acd31(137)=-acd31(92)*acd31(91)
      acd31(138)=-acd31(94)*acd31(93)
      acd31(139)=-acd31(96)*acd31(95)
      acd31(140)=-acd31(98)*acd31(97)
      acd31(141)=-acd31(100)*acd31(99)
      acd31(142)=-acd31(102)*acd31(101)
      acd31(143)=-acd31(104)*acd31(103)
      acd31(144)=-acd31(106)*acd31(105)
      brack=acd31(107)+acd31(108)+acd31(109)+acd31(110)+acd31(111)+2.0_ki*acd31&
      &(112)+acd31(113)+acd31(114)+acd31(115)+acd31(116)+acd31(117)+acd31(118)+&
      &acd31(119)+acd31(120)+acd31(121)+acd31(122)+acd31(123)+acd31(124)+acd31(&
      &125)+acd31(126)+acd31(127)+acd31(128)+acd31(129)+acd31(130)+acd31(131)+a&
      &cd31(132)+acd31(133)+acd31(134)+acd31(135)+acd31(136)+acd31(137)+acd31(1&
      &38)+acd31(139)+acd31(140)+acd31(141)+acd31(142)+acd31(143)+acd31(144)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd31h4
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(61) :: acd31
      complex(ki) :: brack
      acd31(1)=d(iv1,iv2)
      acd31(2)=abb31(28)
      acd31(3)=e2(iv1)
      acd31(4)=spvak1l3(iv2)
      acd31(5)=abb31(40)
      acd31(6)=spvak1l4(iv2)
      acd31(7)=abb31(27)
      acd31(8)=spvak2k1(iv2)
      acd31(9)=abb31(20)
      acd31(10)=spvak2l3(iv2)
      acd31(11)=abb31(31)
      acd31(12)=spvak2l4(iv2)
      acd31(13)=abb31(43)
      acd31(14)=spvak2l5(iv2)
      acd31(15)=abb31(30)
      acd31(16)=spval3k1(iv2)
      acd31(17)=abb31(67)
      acd31(18)=spval3l5(iv2)
      acd31(19)=abb31(74)
      acd31(20)=spval5l3(iv2)
      acd31(21)=abb31(48)
      acd31(22)=spval5l4(iv2)
      acd31(23)=abb31(56)
      acd31(24)=spvak2e1(iv2)
      acd31(25)=abb31(29)
      acd31(26)=spval3e1(iv2)
      acd31(27)=abb31(39)
      acd31(28)=spvae1l3(iv2)
      acd31(29)=abb31(44)
      acd31(30)=spvae1l4(iv2)
      acd31(31)=abb31(68)
      acd31(32)=e2(iv2)
      acd31(33)=spvak1l3(iv1)
      acd31(34)=spvak1l4(iv1)
      acd31(35)=spvak2k1(iv1)
      acd31(36)=spvak2l3(iv1)
      acd31(37)=spvak2l4(iv1)
      acd31(38)=spvak2l5(iv1)
      acd31(39)=spval3k1(iv1)
      acd31(40)=spval3l5(iv1)
      acd31(41)=spval5l3(iv1)
      acd31(42)=spval5l4(iv1)
      acd31(43)=spvak2e1(iv1)
      acd31(44)=spval3e1(iv1)
      acd31(45)=spvae1l3(iv1)
      acd31(46)=spvae1l4(iv1)
      acd31(47)=acd31(4)*acd31(5)
      acd31(48)=acd31(6)*acd31(7)
      acd31(49)=acd31(8)*acd31(9)
      acd31(50)=acd31(10)*acd31(11)
      acd31(51)=acd31(12)*acd31(13)
      acd31(52)=acd31(14)*acd31(15)
      acd31(53)=acd31(16)*acd31(17)
      acd31(54)=acd31(18)*acd31(19)
      acd31(55)=acd31(20)*acd31(21)
      acd31(56)=acd31(22)*acd31(23)
      acd31(57)=acd31(24)*acd31(25)
      acd31(58)=acd31(26)*acd31(27)
      acd31(59)=acd31(28)*acd31(29)
      acd31(60)=acd31(30)*acd31(31)
      acd31(47)=acd31(60)+acd31(59)+acd31(58)+acd31(57)+acd31(56)+acd31(55)+acd&
      &31(54)+acd31(53)+acd31(52)+acd31(51)+acd31(50)+acd31(49)+acd31(48)+acd31&
      &(47)
      acd31(47)=acd31(3)*acd31(47)
      acd31(48)=acd31(33)*acd31(5)
      acd31(49)=acd31(34)*acd31(7)
      acd31(50)=acd31(35)*acd31(9)
      acd31(51)=acd31(36)*acd31(11)
      acd31(52)=acd31(37)*acd31(13)
      acd31(53)=acd31(38)*acd31(15)
      acd31(54)=acd31(39)*acd31(17)
      acd31(55)=acd31(40)*acd31(19)
      acd31(56)=acd31(41)*acd31(21)
      acd31(57)=acd31(42)*acd31(23)
      acd31(58)=acd31(43)*acd31(25)
      acd31(59)=acd31(44)*acd31(27)
      acd31(60)=acd31(45)*acd31(29)
      acd31(61)=acd31(46)*acd31(31)
      acd31(48)=acd31(61)+acd31(60)+acd31(59)+acd31(58)+acd31(57)+acd31(56)+acd&
      &31(55)+acd31(54)+acd31(53)+acd31(52)+acd31(51)+acd31(50)+acd31(49)+acd31&
      &(48)
      acd31(48)=acd31(32)*acd31(48)
      acd31(49)=acd31(2)*acd31(1)
      brack=acd31(47)+acd31(48)+2.0_ki*acd31(49)
   end function brack_3
!---#] function brack_3:
!---#[ function derivative:
   function derivative(mu2,i1,i2) result(numerator)
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd31h4
      implicit none
      complex(ki), intent(in) :: mu2
      integer, intent(in), optional :: i1
      integer, intent(in), optional :: i2
      complex(ki) :: numerator
      complex(ki) :: loc
      integer :: t1
      integer :: deg
      complex(ki), dimension(4), parameter :: Q = (/ (0.0_ki,0.0_ki),(0.0_ki,0.&
      &0_ki),(0.0_ki,0.0_ki),(0.0_ki,0.0_ki)/)
      qshift = -k2
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
   end function derivative
!---#] function derivative:
end module     p2_gg_httbar_d31h4l1d
