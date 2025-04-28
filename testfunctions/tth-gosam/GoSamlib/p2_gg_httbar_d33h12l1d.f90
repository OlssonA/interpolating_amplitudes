module     p2_gg_httbar_d33h12l1d
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity12d33h12l1d.f90
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
      use p2_gg_httbar_abbrevd33h12
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(95) :: acd33
      complex(ki) :: brack
      acd33(1)=dotproduct(k2,qshift)
      acd33(2)=dotproduct(e1,qshift)
      acd33(3)=abb33(17)
      acd33(4)=abb33(16)
      acd33(5)=dotproduct(l4,qshift)
      acd33(6)=abb33(12)
      acd33(7)=dotproduct(qshift,spvak2l4)
      acd33(8)=abb33(10)
      acd33(9)=dotproduct(qshift,spval3k2)
      acd33(10)=abb33(35)
      acd33(11)=dotproduct(qshift,spval3l4)
      acd33(12)=abb33(41)
      acd33(13)=dotproduct(qshift,spval4l3)
      acd33(14)=abb33(27)
      acd33(15)=dotproduct(qshift,spval4l5)
      acd33(16)=abb33(25)
      acd33(17)=dotproduct(qshift,spvak2e2)
      acd33(18)=abb33(28)
      acd33(19)=dotproduct(qshift,spval3e2)
      acd33(20)=abb33(47)
      acd33(21)=dotproduct(qshift,spvae2l3)
      acd33(22)=abb33(46)
      acd33(23)=dotproduct(qshift,spvae2l5)
      acd33(24)=abb33(23)
      acd33(25)=abb33(9)
      acd33(26)=dotproduct(qshift,qshift)
      acd33(27)=abb33(37)
      acd33(28)=abb33(44)
      acd33(29)=abb33(42)
      acd33(30)=abb33(36)
      acd33(31)=abb33(26)
      acd33(32)=abb33(22)
      acd33(33)=abb33(31)
      acd33(34)=dotproduct(qshift,spvak2l3)
      acd33(35)=abb33(13)
      acd33(36)=dotproduct(qshift,spvak2l5)
      acd33(37)=abb33(43)
      acd33(38)=dotproduct(qshift,spval4k2)
      acd33(39)=abb33(33)
      acd33(40)=dotproduct(qshift,spvak1e1)
      acd33(41)=abb33(14)
      acd33(42)=dotproduct(qshift,spvae1k1)
      acd33(43)=abb33(29)
      acd33(44)=dotproduct(qshift,spvak2e1)
      acd33(45)=abb33(76)
      acd33(46)=dotproduct(qshift,spvae1k2)
      acd33(47)=abb33(32)
      acd33(48)=dotproduct(qshift,spvae2k2)
      acd33(49)=abb33(21)
      acd33(50)=dotproduct(qshift,spval3e1)
      acd33(51)=abb33(49)
      acd33(52)=dotproduct(qshift,spvae1l3)
      acd33(53)=abb33(11)
      acd33(54)=dotproduct(qshift,spval4e1)
      acd33(55)=abb33(34)
      acd33(56)=dotproduct(qshift,spvae1l4)
      acd33(57)=abb33(30)
      acd33(58)=dotproduct(qshift,spval4e2)
      acd33(59)=abb33(40)
      acd33(60)=dotproduct(qshift,spvae2l4)
      acd33(61)=abb33(38)
      acd33(62)=dotproduct(qshift,spvae1l5)
      acd33(63)=abb33(24)
      acd33(64)=dotproduct(qshift,spvae1e2)
      acd33(65)=abb33(19)
      acd33(66)=dotproduct(qshift,spvae2e1)
      acd33(67)=abb33(18)
      acd33(68)=abb33(15)
      acd33(69)=acd33(3)*acd33(1)
      acd33(70)=acd33(8)*acd33(7)
      acd33(71)=acd33(10)*acd33(9)
      acd33(72)=acd33(12)*acd33(11)
      acd33(73)=acd33(14)*acd33(13)
      acd33(74)=acd33(16)*acd33(15)
      acd33(75)=acd33(18)*acd33(17)
      acd33(76)=acd33(20)*acd33(19)
      acd33(77)=acd33(22)*acd33(21)
      acd33(78)=acd33(24)*acd33(23)
      acd33(69)=-acd33(25)+acd33(78)+acd33(77)+acd33(76)+acd33(75)+acd33(74)+ac&
      &d33(73)+acd33(72)+acd33(71)+acd33(70)+acd33(69)
      acd33(69)=acd33(2)*acd33(69)
      acd33(70)=-acd33(4)*acd33(1)
      acd33(71)=-acd33(6)*acd33(5)
      acd33(72)=acd33(27)*acd33(26)
      acd33(73)=-acd33(28)*acd33(7)
      acd33(74)=-acd33(29)*acd33(9)
      acd33(75)=-acd33(30)*acd33(11)
      acd33(76)=-acd33(31)*acd33(13)
      acd33(77)=-acd33(32)*acd33(15)
      acd33(78)=-acd33(33)*acd33(17)
      acd33(79)=-acd33(35)*acd33(34)
      acd33(80)=-acd33(37)*acd33(36)
      acd33(81)=-acd33(39)*acd33(38)
      acd33(82)=-acd33(41)*acd33(40)
      acd33(83)=-acd33(43)*acd33(42)
      acd33(84)=-acd33(45)*acd33(44)
      acd33(85)=-acd33(47)*acd33(46)
      acd33(86)=-acd33(49)*acd33(48)
      acd33(87)=-acd33(51)*acd33(50)
      acd33(88)=-acd33(53)*acd33(52)
      acd33(89)=-acd33(55)*acd33(54)
      acd33(90)=-acd33(57)*acd33(56)
      acd33(91)=-acd33(59)*acd33(58)
      acd33(92)=-acd33(61)*acd33(60)
      acd33(93)=-acd33(63)*acd33(62)
      acd33(94)=-acd33(65)*acd33(64)
      acd33(95)=-acd33(67)*acd33(66)
      brack=acd33(68)+acd33(69)+acd33(70)+acd33(71)+acd33(72)+acd33(73)+acd33(7&
      &4)+acd33(75)+acd33(76)+acd33(77)+acd33(78)+acd33(79)+acd33(80)+acd33(81)&
      &+acd33(82)+acd33(83)+acd33(84)+acd33(85)+acd33(86)+acd33(87)+acd33(88)+a&
      &cd33(89)+acd33(90)+acd33(91)+acd33(92)+acd33(93)+acd33(94)+acd33(95)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd33h12
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(106) :: acd33
      complex(ki) :: brack
      acd33(1)=k2(iv1)
      acd33(2)=dotproduct(e1,qshift)
      acd33(3)=abb33(17)
      acd33(4)=abb33(16)
      acd33(5)=l4(iv1)
      acd33(6)=abb33(12)
      acd33(7)=e1(iv1)
      acd33(8)=dotproduct(k2,qshift)
      acd33(9)=dotproduct(qshift,spvak2l4)
      acd33(10)=abb33(10)
      acd33(11)=dotproduct(qshift,spval3k2)
      acd33(12)=abb33(35)
      acd33(13)=dotproduct(qshift,spval3l4)
      acd33(14)=abb33(41)
      acd33(15)=dotproduct(qshift,spval4l3)
      acd33(16)=abb33(27)
      acd33(17)=dotproduct(qshift,spval4l5)
      acd33(18)=abb33(25)
      acd33(19)=dotproduct(qshift,spvak2e2)
      acd33(20)=abb33(28)
      acd33(21)=dotproduct(qshift,spval3e2)
      acd33(22)=abb33(47)
      acd33(23)=dotproduct(qshift,spvae2l3)
      acd33(24)=abb33(46)
      acd33(25)=dotproduct(qshift,spvae2l5)
      acd33(26)=abb33(23)
      acd33(27)=abb33(9)
      acd33(28)=qshift(iv1)
      acd33(29)=abb33(37)
      acd33(30)=spvak2l4(iv1)
      acd33(31)=abb33(44)
      acd33(32)=spval3k2(iv1)
      acd33(33)=abb33(42)
      acd33(34)=spval3l4(iv1)
      acd33(35)=abb33(36)
      acd33(36)=spval4l3(iv1)
      acd33(37)=abb33(26)
      acd33(38)=spval4l5(iv1)
      acd33(39)=abb33(22)
      acd33(40)=spvak2e2(iv1)
      acd33(41)=abb33(31)
      acd33(42)=spval3e2(iv1)
      acd33(43)=spvae2l3(iv1)
      acd33(44)=spvae2l5(iv1)
      acd33(45)=spvak2l3(iv1)
      acd33(46)=abb33(13)
      acd33(47)=spvak2l5(iv1)
      acd33(48)=abb33(43)
      acd33(49)=spval4k2(iv1)
      acd33(50)=abb33(33)
      acd33(51)=spvak1e1(iv1)
      acd33(52)=abb33(14)
      acd33(53)=spvae1k1(iv1)
      acd33(54)=abb33(29)
      acd33(55)=spvak2e1(iv1)
      acd33(56)=abb33(76)
      acd33(57)=spvae1k2(iv1)
      acd33(58)=abb33(32)
      acd33(59)=spvae2k2(iv1)
      acd33(60)=abb33(21)
      acd33(61)=spval3e1(iv1)
      acd33(62)=abb33(49)
      acd33(63)=spvae1l3(iv1)
      acd33(64)=abb33(11)
      acd33(65)=spval4e1(iv1)
      acd33(66)=abb33(34)
      acd33(67)=spvae1l4(iv1)
      acd33(68)=abb33(30)
      acd33(69)=spval4e2(iv1)
      acd33(70)=abb33(40)
      acd33(71)=spvae2l4(iv1)
      acd33(72)=abb33(38)
      acd33(73)=spvae1l5(iv1)
      acd33(74)=abb33(24)
      acd33(75)=spvae1e2(iv1)
      acd33(76)=abb33(19)
      acd33(77)=spvae2e1(iv1)
      acd33(78)=abb33(18)
      acd33(79)=-acd33(3)*acd33(1)
      acd33(80)=-acd33(30)*acd33(10)
      acd33(81)=-acd33(32)*acd33(12)
      acd33(82)=-acd33(34)*acd33(14)
      acd33(83)=-acd33(36)*acd33(16)
      acd33(84)=-acd33(38)*acd33(18)
      acd33(85)=-acd33(40)*acd33(20)
      acd33(86)=-acd33(42)*acd33(22)
      acd33(87)=-acd33(43)*acd33(24)
      acd33(88)=-acd33(44)*acd33(26)
      acd33(79)=acd33(88)+acd33(87)+acd33(86)+acd33(85)+acd33(84)+acd33(83)+acd&
      &33(82)+acd33(81)+acd33(79)+acd33(80)
      acd33(79)=acd33(2)*acd33(79)
      acd33(80)=-acd33(8)*acd33(3)
      acd33(81)=-acd33(9)*acd33(10)
      acd33(82)=-acd33(11)*acd33(12)
      acd33(83)=-acd33(13)*acd33(14)
      acd33(84)=-acd33(15)*acd33(16)
      acd33(85)=-acd33(17)*acd33(18)
      acd33(86)=-acd33(19)*acd33(20)
      acd33(87)=-acd33(21)*acd33(22)
      acd33(88)=-acd33(23)*acd33(24)
      acd33(89)=-acd33(25)*acd33(26)
      acd33(80)=acd33(27)+acd33(89)+acd33(88)+acd33(87)+acd33(86)+acd33(85)+acd&
      &33(84)+acd33(83)+acd33(82)+acd33(81)+acd33(80)
      acd33(80)=acd33(7)*acd33(80)
      acd33(81)=acd33(4)*acd33(1)
      acd33(82)=acd33(6)*acd33(5)
      acd33(83)=acd33(29)*acd33(28)
      acd33(84)=acd33(31)*acd33(30)
      acd33(85)=acd33(33)*acd33(32)
      acd33(86)=acd33(35)*acd33(34)
      acd33(87)=acd33(37)*acd33(36)
      acd33(88)=acd33(39)*acd33(38)
      acd33(89)=acd33(41)*acd33(40)
      acd33(90)=acd33(46)*acd33(45)
      acd33(91)=acd33(48)*acd33(47)
      acd33(92)=acd33(50)*acd33(49)
      acd33(93)=acd33(52)*acd33(51)
      acd33(94)=acd33(54)*acd33(53)
      acd33(95)=acd33(56)*acd33(55)
      acd33(96)=acd33(58)*acd33(57)
      acd33(97)=acd33(60)*acd33(59)
      acd33(98)=acd33(62)*acd33(61)
      acd33(99)=acd33(64)*acd33(63)
      acd33(100)=acd33(66)*acd33(65)
      acd33(101)=acd33(68)*acd33(67)
      acd33(102)=acd33(70)*acd33(69)
      acd33(103)=acd33(72)*acd33(71)
      acd33(104)=acd33(74)*acd33(73)
      acd33(105)=acd33(76)*acd33(75)
      acd33(106)=acd33(78)*acd33(77)
      brack=acd33(79)+acd33(80)+acd33(81)+acd33(82)-2.0_ki*acd33(83)+acd33(84)+&
      &acd33(85)+acd33(86)+acd33(87)+acd33(88)+acd33(89)+acd33(90)+acd33(91)+ac&
      &d33(92)+acd33(93)+acd33(94)+acd33(95)+acd33(96)+acd33(97)+acd33(98)+acd3&
      &3(99)+acd33(100)+acd33(101)+acd33(102)+acd33(103)+acd33(104)+acd33(105)+&
      &acd33(106)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd33h12
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(45) :: acd33
      complex(ki) :: brack
      acd33(1)=d(iv1,iv2)
      acd33(2)=abb33(37)
      acd33(3)=k2(iv1)
      acd33(4)=e1(iv2)
      acd33(5)=abb33(17)
      acd33(6)=k2(iv2)
      acd33(7)=e1(iv1)
      acd33(8)=spvak2l4(iv2)
      acd33(9)=abb33(10)
      acd33(10)=spval3k2(iv2)
      acd33(11)=abb33(35)
      acd33(12)=spval3l4(iv2)
      acd33(13)=abb33(41)
      acd33(14)=spval4l3(iv2)
      acd33(15)=abb33(27)
      acd33(16)=spval4l5(iv2)
      acd33(17)=abb33(25)
      acd33(18)=spvak2e2(iv2)
      acd33(19)=abb33(28)
      acd33(20)=spval3e2(iv2)
      acd33(21)=abb33(47)
      acd33(22)=spvae2l3(iv2)
      acd33(23)=abb33(46)
      acd33(24)=spvae2l5(iv2)
      acd33(25)=abb33(23)
      acd33(26)=spvak2l4(iv1)
      acd33(27)=spval3k2(iv1)
      acd33(28)=spval3l4(iv1)
      acd33(29)=spval4l3(iv1)
      acd33(30)=spval4l5(iv1)
      acd33(31)=spvak2e2(iv1)
      acd33(32)=spval3e2(iv1)
      acd33(33)=spvae2l3(iv1)
      acd33(34)=spvae2l5(iv1)
      acd33(35)=acd33(3)*acd33(5)
      acd33(36)=acd33(26)*acd33(9)
      acd33(37)=acd33(27)*acd33(11)
      acd33(38)=acd33(28)*acd33(13)
      acd33(39)=acd33(29)*acd33(15)
      acd33(40)=acd33(30)*acd33(17)
      acd33(41)=acd33(31)*acd33(19)
      acd33(42)=acd33(32)*acd33(21)
      acd33(43)=acd33(33)*acd33(23)
      acd33(44)=acd33(34)*acd33(25)
      acd33(35)=acd33(44)+acd33(43)+acd33(42)+acd33(41)+acd33(40)+acd33(39)+acd&
      &33(38)+acd33(37)+acd33(36)+acd33(35)
      acd33(35)=acd33(4)*acd33(35)
      acd33(36)=acd33(6)*acd33(5)
      acd33(37)=acd33(8)*acd33(9)
      acd33(38)=acd33(10)*acd33(11)
      acd33(39)=acd33(12)*acd33(13)
      acd33(40)=acd33(14)*acd33(15)
      acd33(41)=acd33(16)*acd33(17)
      acd33(42)=acd33(18)*acd33(19)
      acd33(43)=acd33(20)*acd33(21)
      acd33(44)=acd33(22)*acd33(23)
      acd33(45)=acd33(24)*acd33(25)
      acd33(36)=acd33(45)+acd33(44)+acd33(43)+acd33(42)+acd33(41)+acd33(40)+acd&
      &33(39)+acd33(38)+acd33(37)+acd33(36)
      acd33(36)=acd33(7)*acd33(36)
      acd33(37)=acd33(2)*acd33(1)
      brack=acd33(35)+acd33(36)+2.0_ki*acd33(37)
   end function brack_3
!---#] function brack_3:
!---#[ function derivative:
   function derivative(mu2,i1,i2) result(numerator)
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd33h12
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
      qshift = k2
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
end module     p2_gg_httbar_d33h12l1d
