module     p2_gg_httbar_d28h4l1d
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d28h4l1d.f90
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
      use p2_gg_httbar_abbrevd28h4
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(129) :: acd28
      complex(ki) :: brack
      acd28(1)=dotproduct(k1,qshift)
      acd28(2)=abb28(51)
      acd28(3)=dotproduct(k2,qshift)
      acd28(4)=abb28(15)
      acd28(5)=dotproduct(l4,qshift)
      acd28(6)=abb28(17)
      acd28(7)=dotproduct(e2,qshift)
      acd28(8)=dotproduct(qshift,spvak1k2)
      acd28(9)=abb28(13)
      acd28(10)=dotproduct(qshift,spvak1l3)
      acd28(11)=abb28(55)
      acd28(12)=dotproduct(qshift,spval3k1)
      acd28(13)=abb28(47)
      acd28(14)=dotproduct(qshift,spval3k2)
      acd28(15)=abb28(39)
      acd28(16)=dotproduct(qshift,spval3l4)
      acd28(17)=abb28(45)
      acd28(18)=dotproduct(qshift,spval4k2)
      acd28(19)=abb28(32)
      acd28(20)=dotproduct(qshift,spval4l3)
      acd28(21)=abb28(80)
      acd28(22)=dotproduct(qshift,spval5k1)
      acd28(23)=abb28(50)
      acd28(24)=dotproduct(qshift,spval5k2)
      acd28(25)=abb28(74)
      acd28(26)=dotproduct(qshift,spval5l4)
      acd28(27)=abb28(63)
      acd28(28)=dotproduct(qshift,spvae1k2)
      acd28(29)=abb28(54)
      acd28(30)=dotproduct(qshift,spval3e1)
      acd28(31)=abb28(18)
      acd28(32)=dotproduct(qshift,spvae1l3)
      acd28(33)=abb28(37)
      acd28(34)=dotproduct(qshift,spval5e1)
      acd28(35)=abb28(33)
      acd28(36)=abb28(9)
      acd28(37)=dotproduct(qshift,qshift)
      acd28(38)=abb28(23)
      acd28(39)=abb28(10)
      acd28(40)=abb28(53)
      acd28(41)=abb28(46)
      acd28(42)=abb28(16)
      acd28(43)=abb28(43)
      acd28(44)=abb28(25)
      acd28(45)=abb28(73)
      acd28(46)=abb28(22)
      acd28(47)=abb28(20)
      acd28(48)=abb28(60)
      acd28(49)=abb28(42)
      acd28(50)=dotproduct(qshift,spvak1l4)
      acd28(51)=abb28(48)
      acd28(52)=dotproduct(qshift,spvak2k1)
      acd28(53)=abb28(44)
      acd28(54)=dotproduct(qshift,spvak2l3)
      acd28(55)=abb28(34)
      acd28(56)=dotproduct(qshift,spvak2l4)
      acd28(57)=abb28(52)
      acd28(58)=dotproduct(qshift,spval4k1)
      acd28(59)=abb28(35)
      acd28(60)=dotproduct(qshift,spvak1e1)
      acd28(61)=abb28(19)
      acd28(62)=dotproduct(qshift,spvae1k1)
      acd28(63)=abb28(30)
      acd28(64)=dotproduct(qshift,spvak1e2)
      acd28(65)=abb28(40)
      acd28(66)=dotproduct(qshift,spvae2k1)
      acd28(67)=abb28(38)
      acd28(68)=dotproduct(qshift,spvak2e1)
      acd28(69)=abb28(27)
      acd28(70)=dotproduct(qshift,spvak2e2)
      acd28(71)=abb28(24)
      acd28(72)=dotproduct(qshift,spvae2k2)
      acd28(73)=abb28(12)
      acd28(74)=dotproduct(qshift,spval3e2)
      acd28(75)=abb28(11)
      acd28(76)=dotproduct(qshift,spvae2l3)
      acd28(77)=abb28(69)
      acd28(78)=dotproduct(qshift,spval4e1)
      acd28(79)=abb28(68)
      acd28(80)=dotproduct(qshift,spvae1l4)
      acd28(81)=abb28(61)
      acd28(82)=dotproduct(qshift,spval4e2)
      acd28(83)=abb28(21)
      acd28(84)=dotproduct(qshift,spvae2l4)
      acd28(85)=abb28(56)
      acd28(86)=dotproduct(qshift,spval5e2)
      acd28(87)=abb28(31)
      acd28(88)=dotproduct(qshift,spvae1e2)
      acd28(89)=abb28(28)
      acd28(90)=dotproduct(qshift,spvae2e1)
      acd28(91)=abb28(26)
      acd28(92)=abb28(14)
      acd28(93)=acd28(9)*acd28(8)
      acd28(94)=acd28(11)*acd28(10)
      acd28(95)=acd28(13)*acd28(12)
      acd28(96)=acd28(15)*acd28(14)
      acd28(97)=acd28(17)*acd28(16)
      acd28(98)=acd28(19)*acd28(18)
      acd28(99)=acd28(21)*acd28(20)
      acd28(100)=acd28(23)*acd28(22)
      acd28(101)=acd28(25)*acd28(24)
      acd28(102)=acd28(27)*acd28(26)
      acd28(103)=acd28(29)*acd28(28)
      acd28(104)=acd28(31)*acd28(30)
      acd28(105)=acd28(33)*acd28(32)
      acd28(106)=acd28(35)*acd28(34)
      acd28(93)=-acd28(36)+acd28(106)+acd28(105)+acd28(104)+acd28(103)+acd28(10&
      &2)+acd28(101)+acd28(100)+acd28(99)+acd28(98)+acd28(97)+acd28(96)+acd28(9&
      &5)+acd28(94)+acd28(93)
      acd28(93)=acd28(7)*acd28(93)
      acd28(94)=-acd28(2)*acd28(1)
      acd28(95)=-acd28(4)*acd28(3)
      acd28(96)=-acd28(6)*acd28(5)
      acd28(97)=-acd28(38)*acd28(37)
      acd28(98)=-acd28(39)*acd28(8)
      acd28(99)=-acd28(40)*acd28(10)
      acd28(100)=-acd28(41)*acd28(12)
      acd28(101)=-acd28(42)*acd28(14)
      acd28(102)=-acd28(43)*acd28(16)
      acd28(103)=-acd28(44)*acd28(18)
      acd28(104)=-acd28(45)*acd28(20)
      acd28(105)=-acd28(46)*acd28(22)
      acd28(106)=-acd28(47)*acd28(24)
      acd28(107)=-acd28(48)*acd28(26)
      acd28(108)=-acd28(49)*acd28(28)
      acd28(109)=-acd28(51)*acd28(50)
      acd28(110)=-acd28(53)*acd28(52)
      acd28(111)=-acd28(55)*acd28(54)
      acd28(112)=-acd28(57)*acd28(56)
      acd28(113)=-acd28(59)*acd28(58)
      acd28(114)=-acd28(61)*acd28(60)
      acd28(115)=-acd28(63)*acd28(62)
      acd28(116)=-acd28(65)*acd28(64)
      acd28(117)=-acd28(67)*acd28(66)
      acd28(118)=-acd28(69)*acd28(68)
      acd28(119)=-acd28(71)*acd28(70)
      acd28(120)=-acd28(73)*acd28(72)
      acd28(121)=-acd28(75)*acd28(74)
      acd28(122)=-acd28(77)*acd28(76)
      acd28(123)=-acd28(79)*acd28(78)
      acd28(124)=-acd28(81)*acd28(80)
      acd28(125)=-acd28(83)*acd28(82)
      acd28(126)=-acd28(85)*acd28(84)
      acd28(127)=-acd28(87)*acd28(86)
      acd28(128)=-acd28(89)*acd28(88)
      acd28(129)=-acd28(91)*acd28(90)
      brack=acd28(92)+acd28(93)+acd28(94)+acd28(95)+acd28(96)+acd28(97)+acd28(9&
      &8)+acd28(99)+acd28(100)+acd28(101)+acd28(102)+acd28(103)+acd28(104)+acd2&
      &8(105)+acd28(106)+acd28(107)+acd28(108)+acd28(109)+acd28(110)+acd28(111)&
      &+acd28(112)+acd28(113)+acd28(114)+acd28(115)+acd28(116)+acd28(117)+acd28&
      &(118)+acd28(119)+acd28(120)+acd28(121)+acd28(122)+acd28(123)+acd28(124)+&
      &acd28(125)+acd28(126)+acd28(127)+acd28(128)+acd28(129)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd28h4
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(144) :: acd28
      complex(ki) :: brack
      acd28(1)=k1(iv1)
      acd28(2)=abb28(51)
      acd28(3)=k2(iv1)
      acd28(4)=abb28(15)
      acd28(5)=l4(iv1)
      acd28(6)=abb28(17)
      acd28(7)=e2(iv1)
      acd28(8)=dotproduct(qshift,spvak1k2)
      acd28(9)=abb28(13)
      acd28(10)=dotproduct(qshift,spvak1l3)
      acd28(11)=abb28(55)
      acd28(12)=dotproduct(qshift,spval3k1)
      acd28(13)=abb28(47)
      acd28(14)=dotproduct(qshift,spval3k2)
      acd28(15)=abb28(39)
      acd28(16)=dotproduct(qshift,spval3l4)
      acd28(17)=abb28(45)
      acd28(18)=dotproduct(qshift,spval4k2)
      acd28(19)=abb28(32)
      acd28(20)=dotproduct(qshift,spval4l3)
      acd28(21)=abb28(80)
      acd28(22)=dotproduct(qshift,spval5k1)
      acd28(23)=abb28(50)
      acd28(24)=dotproduct(qshift,spval5k2)
      acd28(25)=abb28(74)
      acd28(26)=dotproduct(qshift,spval5l4)
      acd28(27)=abb28(63)
      acd28(28)=dotproduct(qshift,spvae1k2)
      acd28(29)=abb28(54)
      acd28(30)=dotproduct(qshift,spval3e1)
      acd28(31)=abb28(18)
      acd28(32)=dotproduct(qshift,spvae1l3)
      acd28(33)=abb28(37)
      acd28(34)=dotproduct(qshift,spval5e1)
      acd28(35)=abb28(33)
      acd28(36)=abb28(9)
      acd28(37)=qshift(iv1)
      acd28(38)=abb28(23)
      acd28(39)=spvak1k2(iv1)
      acd28(40)=dotproduct(e2,qshift)
      acd28(41)=abb28(10)
      acd28(42)=spvak1l3(iv1)
      acd28(43)=abb28(53)
      acd28(44)=spval3k1(iv1)
      acd28(45)=abb28(46)
      acd28(46)=spval3k2(iv1)
      acd28(47)=abb28(16)
      acd28(48)=spval3l4(iv1)
      acd28(49)=abb28(43)
      acd28(50)=spval4k2(iv1)
      acd28(51)=abb28(25)
      acd28(52)=spval4l3(iv1)
      acd28(53)=abb28(73)
      acd28(54)=spval5k1(iv1)
      acd28(55)=abb28(22)
      acd28(56)=spval5k2(iv1)
      acd28(57)=abb28(20)
      acd28(58)=spval5l4(iv1)
      acd28(59)=abb28(60)
      acd28(60)=spvae1k2(iv1)
      acd28(61)=abb28(42)
      acd28(62)=spval3e1(iv1)
      acd28(63)=spvae1l3(iv1)
      acd28(64)=spval5e1(iv1)
      acd28(65)=spvak1l4(iv1)
      acd28(66)=abb28(48)
      acd28(67)=spvak2k1(iv1)
      acd28(68)=abb28(44)
      acd28(69)=spvak2l3(iv1)
      acd28(70)=abb28(34)
      acd28(71)=spvak2l4(iv1)
      acd28(72)=abb28(52)
      acd28(73)=spval4k1(iv1)
      acd28(74)=abb28(35)
      acd28(75)=spvak1e1(iv1)
      acd28(76)=abb28(19)
      acd28(77)=spvae1k1(iv1)
      acd28(78)=abb28(30)
      acd28(79)=spvak1e2(iv1)
      acd28(80)=abb28(40)
      acd28(81)=spvae2k1(iv1)
      acd28(82)=abb28(38)
      acd28(83)=spvak2e1(iv1)
      acd28(84)=abb28(27)
      acd28(85)=spvak2e2(iv1)
      acd28(86)=abb28(24)
      acd28(87)=spvae2k2(iv1)
      acd28(88)=abb28(12)
      acd28(89)=spval3e2(iv1)
      acd28(90)=abb28(11)
      acd28(91)=spvae2l3(iv1)
      acd28(92)=abb28(69)
      acd28(93)=spval4e1(iv1)
      acd28(94)=abb28(68)
      acd28(95)=spvae1l4(iv1)
      acd28(96)=abb28(61)
      acd28(97)=spval4e2(iv1)
      acd28(98)=abb28(21)
      acd28(99)=spvae2l4(iv1)
      acd28(100)=abb28(56)
      acd28(101)=spval5e2(iv1)
      acd28(102)=abb28(31)
      acd28(103)=spvae1e2(iv1)
      acd28(104)=abb28(28)
      acd28(105)=spvae2e1(iv1)
      acd28(106)=abb28(26)
      acd28(107)=acd28(39)*acd28(9)
      acd28(108)=acd28(42)*acd28(11)
      acd28(109)=acd28(44)*acd28(13)
      acd28(110)=acd28(46)*acd28(15)
      acd28(111)=acd28(48)*acd28(17)
      acd28(112)=acd28(50)*acd28(19)
      acd28(113)=acd28(52)*acd28(21)
      acd28(114)=acd28(54)*acd28(23)
      acd28(115)=acd28(56)*acd28(25)
      acd28(116)=acd28(58)*acd28(27)
      acd28(117)=acd28(60)*acd28(29)
      acd28(118)=acd28(62)*acd28(31)
      acd28(119)=acd28(63)*acd28(33)
      acd28(120)=acd28(64)*acd28(35)
      acd28(107)=acd28(120)+acd28(119)+acd28(118)+acd28(117)+acd28(116)+acd28(1&
      &15)+acd28(114)+acd28(113)+acd28(112)+acd28(111)+acd28(110)+acd28(109)+ac&
      &d28(107)+acd28(108)
      acd28(107)=acd28(40)*acd28(107)
      acd28(108)=acd28(8)*acd28(9)
      acd28(109)=acd28(10)*acd28(11)
      acd28(110)=acd28(12)*acd28(13)
      acd28(111)=acd28(14)*acd28(15)
      acd28(112)=acd28(16)*acd28(17)
      acd28(113)=acd28(18)*acd28(19)
      acd28(114)=acd28(20)*acd28(21)
      acd28(115)=acd28(22)*acd28(23)
      acd28(116)=acd28(24)*acd28(25)
      acd28(117)=acd28(26)*acd28(27)
      acd28(118)=acd28(28)*acd28(29)
      acd28(119)=acd28(30)*acd28(31)
      acd28(120)=acd28(32)*acd28(33)
      acd28(121)=acd28(34)*acd28(35)
      acd28(108)=-acd28(36)+acd28(121)+acd28(120)+acd28(119)+acd28(118)+acd28(1&
      &17)+acd28(116)+acd28(115)+acd28(114)+acd28(113)+acd28(112)+acd28(111)+ac&
      &d28(110)+acd28(109)+acd28(108)
      acd28(108)=acd28(7)*acd28(108)
      acd28(109)=-acd28(2)*acd28(1)
      acd28(110)=-acd28(4)*acd28(3)
      acd28(111)=-acd28(6)*acd28(5)
      acd28(112)=acd28(38)*acd28(37)
      acd28(113)=-acd28(41)*acd28(39)
      acd28(114)=-acd28(43)*acd28(42)
      acd28(115)=-acd28(45)*acd28(44)
      acd28(116)=-acd28(47)*acd28(46)
      acd28(117)=-acd28(49)*acd28(48)
      acd28(118)=-acd28(51)*acd28(50)
      acd28(119)=-acd28(53)*acd28(52)
      acd28(120)=-acd28(55)*acd28(54)
      acd28(121)=-acd28(57)*acd28(56)
      acd28(122)=-acd28(59)*acd28(58)
      acd28(123)=-acd28(61)*acd28(60)
      acd28(124)=-acd28(66)*acd28(65)
      acd28(125)=-acd28(68)*acd28(67)
      acd28(126)=-acd28(70)*acd28(69)
      acd28(127)=-acd28(72)*acd28(71)
      acd28(128)=-acd28(74)*acd28(73)
      acd28(129)=-acd28(76)*acd28(75)
      acd28(130)=-acd28(78)*acd28(77)
      acd28(131)=-acd28(80)*acd28(79)
      acd28(132)=-acd28(82)*acd28(81)
      acd28(133)=-acd28(84)*acd28(83)
      acd28(134)=-acd28(86)*acd28(85)
      acd28(135)=-acd28(88)*acd28(87)
      acd28(136)=-acd28(90)*acd28(89)
      acd28(137)=-acd28(92)*acd28(91)
      acd28(138)=-acd28(94)*acd28(93)
      acd28(139)=-acd28(96)*acd28(95)
      acd28(140)=-acd28(98)*acd28(97)
      acd28(141)=-acd28(100)*acd28(99)
      acd28(142)=-acd28(102)*acd28(101)
      acd28(143)=-acd28(104)*acd28(103)
      acd28(144)=-acd28(106)*acd28(105)
      brack=acd28(107)+acd28(108)+acd28(109)+acd28(110)+acd28(111)-2.0_ki*acd28&
      &(112)+acd28(113)+acd28(114)+acd28(115)+acd28(116)+acd28(117)+acd28(118)+&
      &acd28(119)+acd28(120)+acd28(121)+acd28(122)+acd28(123)+acd28(124)+acd28(&
      &125)+acd28(126)+acd28(127)+acd28(128)+acd28(129)+acd28(130)+acd28(131)+a&
      &cd28(132)+acd28(133)+acd28(134)+acd28(135)+acd28(136)+acd28(137)+acd28(1&
      &38)+acd28(139)+acd28(140)+acd28(141)+acd28(142)+acd28(143)+acd28(144)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd28h4
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(61) :: acd28
      complex(ki) :: brack
      acd28(1)=d(iv1,iv2)
      acd28(2)=abb28(23)
      acd28(3)=e2(iv1)
      acd28(4)=spvak1k2(iv2)
      acd28(5)=abb28(13)
      acd28(6)=spvak1l3(iv2)
      acd28(7)=abb28(55)
      acd28(8)=spval3k1(iv2)
      acd28(9)=abb28(47)
      acd28(10)=spval3k2(iv2)
      acd28(11)=abb28(39)
      acd28(12)=spval3l4(iv2)
      acd28(13)=abb28(45)
      acd28(14)=spval4k2(iv2)
      acd28(15)=abb28(32)
      acd28(16)=spval4l3(iv2)
      acd28(17)=abb28(80)
      acd28(18)=spval5k1(iv2)
      acd28(19)=abb28(50)
      acd28(20)=spval5k2(iv2)
      acd28(21)=abb28(74)
      acd28(22)=spval5l4(iv2)
      acd28(23)=abb28(63)
      acd28(24)=spvae1k2(iv2)
      acd28(25)=abb28(54)
      acd28(26)=spval3e1(iv2)
      acd28(27)=abb28(18)
      acd28(28)=spvae1l3(iv2)
      acd28(29)=abb28(37)
      acd28(30)=spval5e1(iv2)
      acd28(31)=abb28(33)
      acd28(32)=e2(iv2)
      acd28(33)=spvak1k2(iv1)
      acd28(34)=spvak1l3(iv1)
      acd28(35)=spval3k1(iv1)
      acd28(36)=spval3k2(iv1)
      acd28(37)=spval3l4(iv1)
      acd28(38)=spval4k2(iv1)
      acd28(39)=spval4l3(iv1)
      acd28(40)=spval5k1(iv1)
      acd28(41)=spval5k2(iv1)
      acd28(42)=spval5l4(iv1)
      acd28(43)=spvae1k2(iv1)
      acd28(44)=spval3e1(iv1)
      acd28(45)=spvae1l3(iv1)
      acd28(46)=spval5e1(iv1)
      acd28(47)=acd28(4)*acd28(5)
      acd28(48)=acd28(6)*acd28(7)
      acd28(49)=acd28(8)*acd28(9)
      acd28(50)=acd28(10)*acd28(11)
      acd28(51)=acd28(12)*acd28(13)
      acd28(52)=acd28(14)*acd28(15)
      acd28(53)=acd28(16)*acd28(17)
      acd28(54)=acd28(18)*acd28(19)
      acd28(55)=acd28(20)*acd28(21)
      acd28(56)=acd28(22)*acd28(23)
      acd28(57)=acd28(24)*acd28(25)
      acd28(58)=acd28(26)*acd28(27)
      acd28(59)=acd28(28)*acd28(29)
      acd28(60)=acd28(30)*acd28(31)
      acd28(47)=acd28(60)+acd28(59)+acd28(58)+acd28(57)+acd28(56)+acd28(55)+acd&
      &28(54)+acd28(53)+acd28(52)+acd28(51)+acd28(50)+acd28(49)+acd28(48)+acd28&
      &(47)
      acd28(47)=acd28(3)*acd28(47)
      acd28(48)=acd28(33)*acd28(5)
      acd28(49)=acd28(34)*acd28(7)
      acd28(50)=acd28(35)*acd28(9)
      acd28(51)=acd28(36)*acd28(11)
      acd28(52)=acd28(37)*acd28(13)
      acd28(53)=acd28(38)*acd28(15)
      acd28(54)=acd28(39)*acd28(17)
      acd28(55)=acd28(40)*acd28(19)
      acd28(56)=acd28(41)*acd28(21)
      acd28(57)=acd28(42)*acd28(23)
      acd28(58)=acd28(43)*acd28(25)
      acd28(59)=acd28(44)*acd28(27)
      acd28(60)=acd28(45)*acd28(29)
      acd28(61)=acd28(46)*acd28(31)
      acd28(48)=acd28(61)+acd28(60)+acd28(59)+acd28(58)+acd28(57)+acd28(56)+acd&
      &28(55)+acd28(54)+acd28(53)+acd28(52)+acd28(51)+acd28(50)+acd28(49)+acd28&
      &(48)
      acd28(48)=acd28(32)*acd28(48)
      acd28(49)=acd28(2)*acd28(1)
      brack=acd28(47)+acd28(48)-2.0_ki*acd28(49)
   end function brack_3
!---#] function brack_3:
!---#[ function derivative:
   function derivative(mu2,i1,i2) result(numerator)
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd28h4
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
end module     p2_gg_httbar_d28h4l1d
