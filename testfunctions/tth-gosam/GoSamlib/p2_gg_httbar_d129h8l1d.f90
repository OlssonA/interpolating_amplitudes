module     p2_gg_httbar_d129h8l1d
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d129h8l1d.f90
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
      use p2_gg_httbar_abbrevd129h8
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(64) :: acd129
      complex(ki) :: brack
      acd129(1)=dotproduct(k2,qshift)
      acd129(2)=abb129(21)
      acd129(3)=dotproduct(l3,qshift)
      acd129(4)=abb129(86)
      acd129(5)=dotproduct(l5,qshift)
      acd129(6)=abb129(39)
      acd129(7)=dotproduct(qshift,qshift)
      acd129(8)=abb129(26)
      acd129(9)=dotproduct(qshift,spvak1l3)
      acd129(10)=abb129(179)
      acd129(11)=dotproduct(qshift,spvak1l5)
      acd129(12)=abb129(23)
      acd129(13)=dotproduct(qshift,spvak2k1)
      acd129(14)=abb129(22)
      acd129(15)=dotproduct(qshift,spvak2l3)
      acd129(16)=abb129(18)
      acd129(17)=dotproduct(qshift,spvak2l5)
      acd129(18)=abb129(13)
      acd129(19)=dotproduct(qshift,spval3k1)
      acd129(20)=abb129(161)
      acd129(21)=dotproduct(qshift,spval3k2)
      acd129(22)=abb129(15)
      acd129(23)=dotproduct(qshift,spval3l5)
      acd129(24)=abb129(16)
      acd129(25)=dotproduct(qshift,spval5l3)
      acd129(26)=abb129(17)
      acd129(27)=dotproduct(qshift,spvak2e1)
      acd129(28)=abb129(14)
      acd129(29)=dotproduct(qshift,spvak2e2)
      acd129(30)=abb129(27)
      acd129(31)=dotproduct(qshift,spval3e1)
      acd129(32)=abb129(36)
      acd129(33)=dotproduct(qshift,spvae1l3)
      acd129(34)=abb129(41)
      acd129(35)=dotproduct(qshift,spval3e2)
      acd129(36)=abb129(38)
      acd129(37)=dotproduct(qshift,spvae2l3)
      acd129(38)=abb129(34)
      acd129(39)=dotproduct(qshift,spvae1l5)
      acd129(40)=abb129(20)
      acd129(41)=dotproduct(qshift,spvae2l5)
      acd129(42)=abb129(29)
      acd129(43)=abb129(19)
      acd129(44)=-acd129(2)*acd129(1)
      acd129(45)=-acd129(4)*acd129(3)
      acd129(46)=-acd129(6)*acd129(5)
      acd129(47)=acd129(8)*acd129(7)
      acd129(48)=-acd129(10)*acd129(9)
      acd129(49)=-acd129(12)*acd129(11)
      acd129(50)=-acd129(14)*acd129(13)
      acd129(51)=-acd129(16)*acd129(15)
      acd129(52)=-acd129(18)*acd129(17)
      acd129(53)=-acd129(20)*acd129(19)
      acd129(54)=-acd129(22)*acd129(21)
      acd129(55)=-acd129(24)*acd129(23)
      acd129(56)=-acd129(26)*acd129(25)
      acd129(57)=-acd129(28)*acd129(27)
      acd129(58)=-acd129(30)*acd129(29)
      acd129(59)=-acd129(32)*acd129(31)
      acd129(60)=-acd129(34)*acd129(33)
      acd129(61)=-acd129(36)*acd129(35)
      acd129(62)=-acd129(38)*acd129(37)
      acd129(63)=-acd129(40)*acd129(39)
      acd129(64)=-acd129(42)*acd129(41)
      brack=acd129(43)+acd129(44)+acd129(45)+acd129(46)+acd129(47)+acd129(48)+a&
      &cd129(49)+acd129(50)+acd129(51)+acd129(52)+acd129(53)+acd129(54)+acd129(&
      &55)+acd129(56)+acd129(57)+acd129(58)+acd129(59)+acd129(60)+acd129(61)+ac&
      &d129(62)+acd129(63)+acd129(64)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd129h8
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(63) :: acd129
      complex(ki) :: brack
      acd129(1)=k2(iv1)
      acd129(2)=abb129(21)
      acd129(3)=l3(iv1)
      acd129(4)=abb129(86)
      acd129(5)=l5(iv1)
      acd129(6)=abb129(39)
      acd129(7)=qshift(iv1)
      acd129(8)=abb129(26)
      acd129(9)=spvak1l3(iv1)
      acd129(10)=abb129(179)
      acd129(11)=spvak1l5(iv1)
      acd129(12)=abb129(23)
      acd129(13)=spvak2k1(iv1)
      acd129(14)=abb129(22)
      acd129(15)=spvak2l3(iv1)
      acd129(16)=abb129(18)
      acd129(17)=spvak2l5(iv1)
      acd129(18)=abb129(13)
      acd129(19)=spval3k1(iv1)
      acd129(20)=abb129(161)
      acd129(21)=spval3k2(iv1)
      acd129(22)=abb129(15)
      acd129(23)=spval3l5(iv1)
      acd129(24)=abb129(16)
      acd129(25)=spval5l3(iv1)
      acd129(26)=abb129(17)
      acd129(27)=spvak2e1(iv1)
      acd129(28)=abb129(14)
      acd129(29)=spvak2e2(iv1)
      acd129(30)=abb129(27)
      acd129(31)=spval3e1(iv1)
      acd129(32)=abb129(36)
      acd129(33)=spvae1l3(iv1)
      acd129(34)=abb129(41)
      acd129(35)=spval3e2(iv1)
      acd129(36)=abb129(38)
      acd129(37)=spvae2l3(iv1)
      acd129(38)=abb129(34)
      acd129(39)=spvae1l5(iv1)
      acd129(40)=abb129(20)
      acd129(41)=spvae2l5(iv1)
      acd129(42)=abb129(29)
      acd129(43)=acd129(2)*acd129(1)
      acd129(44)=acd129(4)*acd129(3)
      acd129(45)=acd129(6)*acd129(5)
      acd129(46)=acd129(8)*acd129(7)
      acd129(47)=acd129(10)*acd129(9)
      acd129(48)=acd129(12)*acd129(11)
      acd129(49)=acd129(14)*acd129(13)
      acd129(50)=acd129(16)*acd129(15)
      acd129(51)=acd129(18)*acd129(17)
      acd129(52)=acd129(20)*acd129(19)
      acd129(53)=acd129(22)*acd129(21)
      acd129(54)=acd129(24)*acd129(23)
      acd129(55)=acd129(26)*acd129(25)
      acd129(56)=acd129(28)*acd129(27)
      acd129(57)=acd129(30)*acd129(29)
      acd129(58)=acd129(32)*acd129(31)
      acd129(59)=acd129(34)*acd129(33)
      acd129(60)=acd129(36)*acd129(35)
      acd129(61)=acd129(38)*acd129(37)
      acd129(62)=acd129(40)*acd129(39)
      acd129(63)=acd129(42)*acd129(41)
      brack=acd129(43)+acd129(44)+acd129(45)-2.0_ki*acd129(46)+acd129(47)+acd12&
      &9(48)+acd129(49)+acd129(50)+acd129(51)+acd129(52)+acd129(53)+acd129(54)+&
      &acd129(55)+acd129(56)+acd129(57)+acd129(58)+acd129(59)+acd129(60)+acd129&
      &(61)+acd129(62)+acd129(63)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd129h8
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(3) :: acd129
      complex(ki) :: brack
      acd129(1)=d(iv1,iv2)
      acd129(2)=abb129(26)
      brack=2.0_ki*acd129(2)*acd129(1)
   end function brack_3
!---#] function brack_3:
!---#[ function derivative:
   function derivative(mu2,i1,i2) result(numerator)
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd129h8
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
      qshift = -k5
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
end module     p2_gg_httbar_d129h8l1d
