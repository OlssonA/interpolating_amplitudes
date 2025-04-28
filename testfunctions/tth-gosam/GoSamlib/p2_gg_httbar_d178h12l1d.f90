module     p2_gg_httbar_d178h12l1d
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity12d178h12l1d.f90
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
      use p2_gg_httbar_abbrevd178h12
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(88) :: acd178
      complex(ki) :: brack
      acd178(1)=dotproduct(k2,qshift)
      acd178(2)=dotproduct(e1,qshift)
      acd178(3)=abb178(62)
      acd178(4)=abb178(50)
      acd178(5)=dotproduct(l4,qshift)
      acd178(6)=abb178(45)
      acd178(7)=abb178(18)
      acd178(8)=dotproduct(qshift,spvak1l4)
      acd178(9)=abb178(25)
      acd178(10)=dotproduct(qshift,spvak2k1)
      acd178(11)=abb178(17)
      acd178(12)=dotproduct(qshift,spvak2l4)
      acd178(13)=abb178(20)
      acd178(14)=dotproduct(qshift,spvak2l5)
      acd178(15)=abb178(37)
      acd178(16)=dotproduct(qshift,spval5l4)
      acd178(17)=abb178(27)
      acd178(18)=dotproduct(qshift,spvak2e2)
      acd178(19)=abb178(56)
      acd178(20)=dotproduct(qshift,spvae2l4)
      acd178(21)=abb178(80)
      acd178(22)=abb178(15)
      acd178(23)=dotproduct(qshift,qshift)
      acd178(24)=abb178(83)
      acd178(25)=abb178(23)
      acd178(26)=abb178(11)
      acd178(27)=abb178(29)
      acd178(28)=abb178(54)
      acd178(29)=dotproduct(qshift,spvak1k2)
      acd178(30)=abb178(19)
      acd178(31)=dotproduct(qshift,spval4k1)
      acd178(32)=abb178(21)
      acd178(33)=dotproduct(qshift,spval4k2)
      acd178(34)=abb178(42)
      acd178(35)=dotproduct(qshift,spval4l5)
      acd178(36)=abb178(24)
      acd178(37)=dotproduct(qshift,spval5k2)
      acd178(38)=abb178(166)
      acd178(39)=dotproduct(qshift,spvak1e1)
      acd178(40)=abb178(28)
      acd178(41)=dotproduct(qshift,spvae1k1)
      acd178(42)=abb178(22)
      acd178(43)=dotproduct(qshift,spvak2e1)
      acd178(44)=abb178(16)
      acd178(45)=dotproduct(qshift,spvae1k2)
      acd178(46)=abb178(48)
      acd178(47)=dotproduct(qshift,spvae2k2)
      acd178(48)=abb178(12)
      acd178(49)=dotproduct(qshift,spval4e1)
      acd178(50)=abb178(46)
      acd178(51)=dotproduct(qshift,spvae1l4)
      acd178(52)=abb178(33)
      acd178(53)=dotproduct(qshift,spval4e2)
      acd178(54)=abb178(31)
      acd178(55)=dotproduct(qshift,spval5e1)
      acd178(56)=abb178(154)
      acd178(57)=dotproduct(qshift,spvae1l5)
      acd178(58)=abb178(67)
      acd178(59)=dotproduct(qshift,spvae1e2)
      acd178(60)=abb178(52)
      acd178(61)=dotproduct(qshift,spvae2e1)
      acd178(62)=abb178(14)
      acd178(63)=abb178(13)
      acd178(64)=acd178(3)*acd178(1)
      acd178(65)=acd178(6)*acd178(5)
      acd178(66)=acd178(9)*acd178(8)
      acd178(67)=acd178(11)*acd178(10)
      acd178(68)=acd178(13)*acd178(12)
      acd178(69)=acd178(15)*acd178(14)
      acd178(70)=acd178(17)*acd178(16)
      acd178(71)=acd178(19)*acd178(18)
      acd178(72)=acd178(21)*acd178(20)
      acd178(64)=-acd178(22)+acd178(72)+acd178(71)+acd178(70)+acd178(69)+acd178&
      &(68)+acd178(67)+acd178(66)+acd178(65)+acd178(64)
      acd178(64)=acd178(2)*acd178(64)
      acd178(65)=-acd178(4)*acd178(1)
      acd178(66)=-acd178(7)*acd178(5)
      acd178(67)=acd178(24)*acd178(23)
      acd178(68)=-acd178(25)*acd178(8)
      acd178(69)=-acd178(26)*acd178(12)
      acd178(70)=-acd178(27)*acd178(16)
      acd178(71)=-acd178(28)*acd178(20)
      acd178(72)=-acd178(30)*acd178(29)
      acd178(73)=-acd178(32)*acd178(31)
      acd178(74)=-acd178(34)*acd178(33)
      acd178(75)=-acd178(36)*acd178(35)
      acd178(76)=-acd178(38)*acd178(37)
      acd178(77)=-acd178(40)*acd178(39)
      acd178(78)=-acd178(42)*acd178(41)
      acd178(79)=-acd178(44)*acd178(43)
      acd178(80)=-acd178(46)*acd178(45)
      acd178(81)=-acd178(48)*acd178(47)
      acd178(82)=-acd178(50)*acd178(49)
      acd178(83)=-acd178(52)*acd178(51)
      acd178(84)=-acd178(54)*acd178(53)
      acd178(85)=acd178(56)*acd178(55)
      acd178(86)=-acd178(58)*acd178(57)
      acd178(87)=-acd178(60)*acd178(59)
      acd178(88)=-acd178(62)*acd178(61)
      brack=acd178(63)+acd178(64)+acd178(65)+acd178(66)+acd178(67)+acd178(68)+a&
      &cd178(69)+acd178(70)+acd178(71)+acd178(72)+acd178(73)+acd178(74)+acd178(&
      &75)+acd178(76)+acd178(77)+acd178(78)+acd178(79)+acd178(80)+acd178(81)+ac&
      &d178(82)+acd178(83)+acd178(84)+acd178(85)+acd178(86)+acd178(87)+acd178(8&
      &8)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd178h12
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(98) :: acd178
      complex(ki) :: brack
      acd178(1)=k2(iv1)
      acd178(2)=dotproduct(e1,qshift)
      acd178(3)=abb178(62)
      acd178(4)=abb178(50)
      acd178(5)=l4(iv1)
      acd178(6)=abb178(45)
      acd178(7)=abb178(18)
      acd178(8)=e1(iv1)
      acd178(9)=dotproduct(k2,qshift)
      acd178(10)=dotproduct(l4,qshift)
      acd178(11)=dotproduct(qshift,spvak1l4)
      acd178(12)=abb178(25)
      acd178(13)=dotproduct(qshift,spvak2k1)
      acd178(14)=abb178(17)
      acd178(15)=dotproduct(qshift,spvak2l4)
      acd178(16)=abb178(20)
      acd178(17)=dotproduct(qshift,spvak2l5)
      acd178(18)=abb178(37)
      acd178(19)=dotproduct(qshift,spval5l4)
      acd178(20)=abb178(27)
      acd178(21)=dotproduct(qshift,spvak2e2)
      acd178(22)=abb178(56)
      acd178(23)=dotproduct(qshift,spvae2l4)
      acd178(24)=abb178(80)
      acd178(25)=abb178(15)
      acd178(26)=qshift(iv1)
      acd178(27)=abb178(83)
      acd178(28)=spvak1l4(iv1)
      acd178(29)=abb178(23)
      acd178(30)=spvak2k1(iv1)
      acd178(31)=spvak2l4(iv1)
      acd178(32)=abb178(11)
      acd178(33)=spvak2l5(iv1)
      acd178(34)=spval5l4(iv1)
      acd178(35)=abb178(29)
      acd178(36)=spvak2e2(iv1)
      acd178(37)=spvae2l4(iv1)
      acd178(38)=abb178(54)
      acd178(39)=spvak1k2(iv1)
      acd178(40)=abb178(19)
      acd178(41)=spval4k1(iv1)
      acd178(42)=abb178(21)
      acd178(43)=spval4k2(iv1)
      acd178(44)=abb178(42)
      acd178(45)=spval4l5(iv1)
      acd178(46)=abb178(24)
      acd178(47)=spval5k2(iv1)
      acd178(48)=abb178(166)
      acd178(49)=spvak1e1(iv1)
      acd178(50)=abb178(28)
      acd178(51)=spvae1k1(iv1)
      acd178(52)=abb178(22)
      acd178(53)=spvak2e1(iv1)
      acd178(54)=abb178(16)
      acd178(55)=spvae1k2(iv1)
      acd178(56)=abb178(48)
      acd178(57)=spvae2k2(iv1)
      acd178(58)=abb178(12)
      acd178(59)=spval4e1(iv1)
      acd178(60)=abb178(46)
      acd178(61)=spvae1l4(iv1)
      acd178(62)=abb178(33)
      acd178(63)=spval4e2(iv1)
      acd178(64)=abb178(31)
      acd178(65)=spval5e1(iv1)
      acd178(66)=abb178(154)
      acd178(67)=spvae1l5(iv1)
      acd178(68)=abb178(67)
      acd178(69)=spvae1e2(iv1)
      acd178(70)=abb178(52)
      acd178(71)=spvae2e1(iv1)
      acd178(72)=abb178(14)
      acd178(73)=acd178(3)*acd178(1)
      acd178(74)=acd178(6)*acd178(5)
      acd178(75)=acd178(28)*acd178(12)
      acd178(76)=acd178(31)*acd178(16)
      acd178(77)=acd178(34)*acd178(20)
      acd178(78)=acd178(37)*acd178(24)
      acd178(79)=acd178(30)*acd178(14)
      acd178(80)=acd178(33)*acd178(18)
      acd178(81)=acd178(36)*acd178(22)
      acd178(73)=acd178(81)+acd178(80)+acd178(79)+acd178(78)+acd178(77)+acd178(&
      &76)+acd178(75)+acd178(73)+acd178(74)
      acd178(73)=acd178(2)*acd178(73)
      acd178(74)=acd178(9)*acd178(3)
      acd178(75)=acd178(10)*acd178(6)
      acd178(76)=acd178(11)*acd178(12)
      acd178(77)=acd178(13)*acd178(14)
      acd178(78)=acd178(15)*acd178(16)
      acd178(79)=acd178(17)*acd178(18)
      acd178(80)=acd178(19)*acd178(20)
      acd178(81)=acd178(21)*acd178(22)
      acd178(82)=acd178(23)*acd178(24)
      acd178(74)=-acd178(25)+acd178(82)+acd178(81)+acd178(80)+acd178(79)+acd178&
      &(78)+acd178(77)+acd178(76)+acd178(75)+acd178(74)
      acd178(74)=acd178(8)*acd178(74)
      acd178(75)=-acd178(4)*acd178(1)
      acd178(76)=-acd178(7)*acd178(5)
      acd178(77)=acd178(27)*acd178(26)
      acd178(78)=-acd178(29)*acd178(28)
      acd178(79)=-acd178(32)*acd178(31)
      acd178(80)=-acd178(35)*acd178(34)
      acd178(81)=-acd178(38)*acd178(37)
      acd178(82)=-acd178(40)*acd178(39)
      acd178(83)=-acd178(42)*acd178(41)
      acd178(84)=-acd178(44)*acd178(43)
      acd178(85)=-acd178(46)*acd178(45)
      acd178(86)=-acd178(48)*acd178(47)
      acd178(87)=-acd178(50)*acd178(49)
      acd178(88)=-acd178(52)*acd178(51)
      acd178(89)=-acd178(54)*acd178(53)
      acd178(90)=-acd178(56)*acd178(55)
      acd178(91)=-acd178(58)*acd178(57)
      acd178(92)=-acd178(60)*acd178(59)
      acd178(93)=-acd178(62)*acd178(61)
      acd178(94)=-acd178(64)*acd178(63)
      acd178(95)=acd178(66)*acd178(65)
      acd178(96)=-acd178(68)*acd178(67)
      acd178(97)=-acd178(70)*acd178(69)
      acd178(98)=-acd178(72)*acd178(71)
      brack=acd178(73)+acd178(74)+acd178(75)+acd178(76)+2.0_ki*acd178(77)+acd17&
      &8(78)+acd178(79)+acd178(80)+acd178(81)+acd178(82)+acd178(83)+acd178(84)+&
      &acd178(85)+acd178(86)+acd178(87)+acd178(88)+acd178(89)+acd178(90)+acd178&
      &(91)+acd178(92)+acd178(93)+acd178(94)+acd178(95)+acd178(96)+acd178(97)+a&
      &cd178(98)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd178h12
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(41) :: acd178
      complex(ki) :: brack
      acd178(1)=d(iv1,iv2)
      acd178(2)=abb178(83)
      acd178(3)=k2(iv1)
      acd178(4)=e1(iv2)
      acd178(5)=abb178(62)
      acd178(6)=k2(iv2)
      acd178(7)=e1(iv1)
      acd178(8)=l4(iv1)
      acd178(9)=abb178(45)
      acd178(10)=l4(iv2)
      acd178(11)=spvak1l4(iv2)
      acd178(12)=abb178(25)
      acd178(13)=spvak2k1(iv2)
      acd178(14)=abb178(17)
      acd178(15)=spvak2l4(iv2)
      acd178(16)=abb178(20)
      acd178(17)=spvak2l5(iv2)
      acd178(18)=abb178(37)
      acd178(19)=spval5l4(iv2)
      acd178(20)=abb178(27)
      acd178(21)=spvak2e2(iv2)
      acd178(22)=abb178(56)
      acd178(23)=spvae2l4(iv2)
      acd178(24)=abb178(80)
      acd178(25)=spvak1l4(iv1)
      acd178(26)=spvak2k1(iv1)
      acd178(27)=spvak2l4(iv1)
      acd178(28)=spvak2l5(iv1)
      acd178(29)=spval5l4(iv1)
      acd178(30)=spvak2e2(iv1)
      acd178(31)=spvae2l4(iv1)
      acd178(32)=acd178(3)*acd178(5)
      acd178(33)=acd178(8)*acd178(9)
      acd178(34)=acd178(25)*acd178(12)
      acd178(35)=acd178(26)*acd178(14)
      acd178(36)=acd178(27)*acd178(16)
      acd178(37)=acd178(28)*acd178(18)
      acd178(38)=acd178(29)*acd178(20)
      acd178(39)=acd178(30)*acd178(22)
      acd178(40)=acd178(31)*acd178(24)
      acd178(32)=acd178(40)+acd178(39)+acd178(38)+acd178(37)+acd178(36)+acd178(&
      &35)+acd178(34)+acd178(33)+acd178(32)
      acd178(32)=acd178(4)*acd178(32)
      acd178(33)=acd178(6)*acd178(5)
      acd178(34)=acd178(10)*acd178(9)
      acd178(35)=acd178(11)*acd178(12)
      acd178(36)=acd178(13)*acd178(14)
      acd178(37)=acd178(15)*acd178(16)
      acd178(38)=acd178(17)*acd178(18)
      acd178(39)=acd178(19)*acd178(20)
      acd178(40)=acd178(21)*acd178(22)
      acd178(41)=acd178(23)*acd178(24)
      acd178(33)=acd178(41)+acd178(40)+acd178(39)+acd178(38)+acd178(37)+acd178(&
      &36)+acd178(35)+acd178(34)+acd178(33)
      acd178(33)=acd178(7)*acd178(33)
      acd178(34)=acd178(2)*acd178(1)
      brack=acd178(32)+acd178(33)+2.0_ki*acd178(34)
   end function brack_3
!---#] function brack_3:
!---#[ function derivative:
   function derivative(mu2,i1,i2) result(numerator)
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd178h12
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
      qshift = -k3-k4-k5
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
end module     p2_gg_httbar_d178h12l1d
