module     p2_gg_httbar_d12h12l1d
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity12d12h12l1d.f90
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
      use p2_gg_httbar_abbrevd12h12
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(38) :: acd12
      complex(ki) :: brack
      acd12(1)=dotproduct(k2,qshift)
      acd12(2)=dotproduct(qshift,spvak2l3)
      acd12(3)=abb12(11)
      acd12(4)=dotproduct(qshift,spvak2l4)
      acd12(5)=abb12(17)
      acd12(6)=dotproduct(qshift,spvak2l5)
      acd12(7)=abb12(22)
      acd12(8)=abb12(19)
      acd12(9)=abb12(9)
      acd12(10)=dotproduct(qshift,spval3k2)
      acd12(11)=abb12(21)
      acd12(12)=abb12(16)
      acd12(13)=abb12(13)
      acd12(14)=dotproduct(qshift,spvak1l3)
      acd12(15)=dotproduct(qshift,spvak2k1)
      acd12(16)=abb12(12)
      acd12(17)=abb12(14)
      acd12(18)=dotproduct(qshift,spvak1l4)
      acd12(19)=abb12(28)
      acd12(20)=dotproduct(qshift,spvak1l5)
      acd12(21)=abb12(27)
      acd12(22)=abb12(18)
      acd12(23)=dotproduct(qshift,spval3k1)
      acd12(24)=abb12(10)
      acd12(25)=abb12(26)
      acd12(26)=abb12(25)
      acd12(27)=abb12(20)
      acd12(28)=abb12(15)
      acd12(29)=acd12(3)*acd12(2)
      acd12(30)=acd12(5)*acd12(4)
      acd12(31)=acd12(7)*acd12(6)
      acd12(29)=-acd12(8)+acd12(31)+acd12(30)+acd12(29)
      acd12(29)=acd12(1)*acd12(29)
      acd12(30)=acd12(16)*acd12(14)
      acd12(31)=acd12(19)*acd12(18)
      acd12(32)=acd12(21)*acd12(20)
      acd12(30)=-acd12(22)+acd12(32)+acd12(31)+acd12(30)
      acd12(30)=acd12(15)*acd12(30)
      acd12(31)=acd12(11)*acd12(4)
      acd12(31)=-acd12(27)+acd12(31)
      acd12(31)=acd12(10)*acd12(31)
      acd12(32)=-acd12(11)*acd12(18)
      acd12(32)=-acd12(25)+acd12(32)
      acd12(32)=acd12(23)*acd12(32)
      acd12(33)=-acd12(9)*acd12(2)
      acd12(34)=-acd12(12)*acd12(4)
      acd12(35)=-acd12(13)*acd12(6)
      acd12(36)=-acd12(17)*acd12(14)
      acd12(37)=-acd12(24)*acd12(18)
      acd12(38)=-acd12(26)*acd12(20)
      brack=acd12(28)+acd12(29)+acd12(30)+acd12(31)+acd12(32)+acd12(33)+acd12(3&
      &4)+acd12(35)+acd12(36)+acd12(37)+acd12(38)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd12h12
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(50) :: acd12
      complex(ki) :: brack
      acd12(1)=k2(iv1)
      acd12(2)=dotproduct(qshift,spvak2l3)
      acd12(3)=abb12(11)
      acd12(4)=dotproduct(qshift,spvak2l4)
      acd12(5)=abb12(17)
      acd12(6)=dotproduct(qshift,spvak2l5)
      acd12(7)=abb12(22)
      acd12(8)=abb12(19)
      acd12(9)=spvak2l3(iv1)
      acd12(10)=dotproduct(k2,qshift)
      acd12(11)=abb12(9)
      acd12(12)=spvak2l4(iv1)
      acd12(13)=dotproduct(qshift,spval3k2)
      acd12(14)=abb12(21)
      acd12(15)=abb12(16)
      acd12(16)=spvak2l5(iv1)
      acd12(17)=abb12(13)
      acd12(18)=spvak1l3(iv1)
      acd12(19)=dotproduct(qshift,spvak2k1)
      acd12(20)=abb12(12)
      acd12(21)=abb12(14)
      acd12(22)=spvak2k1(iv1)
      acd12(23)=dotproduct(qshift,spvak1l3)
      acd12(24)=dotproduct(qshift,spvak1l4)
      acd12(25)=abb12(28)
      acd12(26)=dotproduct(qshift,spvak1l5)
      acd12(27)=abb12(27)
      acd12(28)=abb12(18)
      acd12(29)=spvak1l4(iv1)
      acd12(30)=dotproduct(qshift,spval3k1)
      acd12(31)=abb12(10)
      acd12(32)=spval3k1(iv1)
      acd12(33)=abb12(26)
      acd12(34)=spvak1l5(iv1)
      acd12(35)=abb12(25)
      acd12(36)=spval3k2(iv1)
      acd12(37)=abb12(20)
      acd12(38)=acd12(32)*acd12(24)
      acd12(39)=-acd12(36)*acd12(4)
      acd12(40)=-acd12(13)*acd12(12)
      acd12(41)=acd12(30)*acd12(29)
      acd12(38)=acd12(41)+acd12(40)+acd12(39)+acd12(38)
      acd12(38)=acd12(14)*acd12(38)
      acd12(39)=-acd12(2)*acd12(3)
      acd12(40)=-acd12(6)*acd12(7)
      acd12(39)=acd12(8)+acd12(40)+acd12(39)
      acd12(39)=acd12(1)*acd12(39)
      acd12(40)=-acd12(23)*acd12(20)
      acd12(41)=-acd12(26)*acd12(27)
      acd12(40)=acd12(28)+acd12(41)+acd12(40)
      acd12(40)=acd12(22)*acd12(40)
      acd12(41)=-acd12(12)*acd12(10)
      acd12(42)=-acd12(4)*acd12(1)
      acd12(41)=acd12(41)+acd12(42)
      acd12(41)=acd12(5)*acd12(41)
      acd12(42)=-acd12(29)*acd12(19)
      acd12(43)=-acd12(24)*acd12(22)
      acd12(42)=acd12(42)+acd12(43)
      acd12(42)=acd12(25)*acd12(42)
      acd12(43)=-acd12(3)*acd12(10)
      acd12(43)=acd12(11)+acd12(43)
      acd12(43)=acd12(9)*acd12(43)
      acd12(44)=-acd12(7)*acd12(10)
      acd12(44)=acd12(17)+acd12(44)
      acd12(44)=acd12(16)*acd12(44)
      acd12(45)=-acd12(20)*acd12(19)
      acd12(45)=acd12(21)+acd12(45)
      acd12(45)=acd12(18)*acd12(45)
      acd12(46)=-acd12(27)*acd12(19)
      acd12(46)=acd12(35)+acd12(46)
      acd12(46)=acd12(34)*acd12(46)
      acd12(47)=acd12(15)*acd12(12)
      acd12(48)=acd12(31)*acd12(29)
      acd12(49)=acd12(33)*acd12(32)
      acd12(50)=acd12(37)*acd12(36)
      brack=acd12(38)+acd12(39)+acd12(40)+acd12(41)+acd12(42)+acd12(43)+acd12(4&
      &4)+acd12(45)+acd12(46)+acd12(47)+acd12(48)+acd12(49)+acd12(50)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd12h12
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(34) :: acd12
      complex(ki) :: brack
      acd12(1)=k2(iv1)
      acd12(2)=spvak2l3(iv2)
      acd12(3)=abb12(11)
      acd12(4)=spvak2l4(iv2)
      acd12(5)=abb12(17)
      acd12(6)=spvak2l5(iv2)
      acd12(7)=abb12(22)
      acd12(8)=k2(iv2)
      acd12(9)=spvak2l3(iv1)
      acd12(10)=spvak2l4(iv1)
      acd12(11)=spvak2l5(iv1)
      acd12(12)=spval3k2(iv2)
      acd12(13)=abb12(21)
      acd12(14)=spval3k2(iv1)
      acd12(15)=spvak1l3(iv1)
      acd12(16)=spvak2k1(iv2)
      acd12(17)=abb12(12)
      acd12(18)=spvak1l3(iv2)
      acd12(19)=spvak2k1(iv1)
      acd12(20)=spvak1l4(iv2)
      acd12(21)=abb12(28)
      acd12(22)=spvak1l5(iv2)
      acd12(23)=abb12(27)
      acd12(24)=spvak1l4(iv1)
      acd12(25)=spvak1l5(iv1)
      acd12(26)=spval3k1(iv2)
      acd12(27)=spval3k1(iv1)
      acd12(28)=acd12(12)*acd12(10)
      acd12(29)=acd12(14)*acd12(4)
      acd12(30)=-acd12(26)*acd12(24)
      acd12(31)=-acd12(27)*acd12(20)
      acd12(28)=acd12(31)+acd12(30)+acd12(29)+acd12(28)
      acd12(28)=acd12(13)*acd12(28)
      acd12(29)=acd12(5)*acd12(4)
      acd12(30)=acd12(2)*acd12(3)
      acd12(31)=acd12(6)*acd12(7)
      acd12(29)=acd12(31)+acd12(30)+acd12(29)
      acd12(29)=acd12(1)*acd12(29)
      acd12(30)=acd12(10)*acd12(5)
      acd12(31)=acd12(9)*acd12(3)
      acd12(32)=acd12(11)*acd12(7)
      acd12(30)=acd12(32)+acd12(31)+acd12(30)
      acd12(30)=acd12(8)*acd12(30)
      acd12(31)=acd12(24)*acd12(21)
      acd12(32)=acd12(15)*acd12(17)
      acd12(33)=acd12(25)*acd12(23)
      acd12(31)=acd12(33)+acd12(32)+acd12(31)
      acd12(31)=acd12(16)*acd12(31)
      acd12(32)=acd12(21)*acd12(20)
      acd12(33)=acd12(18)*acd12(17)
      acd12(34)=acd12(22)*acd12(23)
      acd12(32)=acd12(34)+acd12(33)+acd12(32)
      acd12(32)=acd12(19)*acd12(32)
      brack=acd12(28)+acd12(29)+acd12(30)+acd12(31)+acd12(32)
   end function brack_3
!---#] function brack_3:
!---#[ function derivative:
   function derivative(mu2,i1,i2) result(numerator)
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd12h12
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
      qshift = k4
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
end module     p2_gg_httbar_d12h12l1d
