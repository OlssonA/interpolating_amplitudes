module     p2_gg_httbar_d34h0l1d
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d34h0l1d.f90
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
      use p2_gg_httbar_abbrevd34h0
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(38) :: acd34
      complex(ki) :: brack
      acd34(1)=dotproduct(qshift,spvak2e1)
      acd34(2)=dotproduct(qshift,spvae1k2)
      acd34(3)=abb34(13)
      acd34(4)=dotproduct(qshift,spvae1l3)
      acd34(5)=abb34(20)
      acd34(6)=abb34(10)
      acd34(7)=dotproduct(qshift,spval5e1)
      acd34(8)=abb34(27)
      acd34(9)=dotproduct(qshift,spvae2e1)
      acd34(10)=abb34(23)
      acd34(11)=abb34(25)
      acd34(12)=abb34(19)
      acd34(13)=abb34(24)
      acd34(14)=abb34(18)
      acd34(15)=abb34(36)
      acd34(16)=abb34(14)
      acd34(17)=dotproduct(qshift,spval3e1)
      acd34(18)=dotproduct(qshift,spvae1l5)
      acd34(19)=abb34(48)
      acd34(20)=dotproduct(qshift,spvae1e2)
      acd34(21)=abb34(50)
      acd34(22)=abb34(12)
      acd34(23)=dotproduct(qshift,spval4e1)
      acd34(24)=abb34(52)
      acd34(25)=abb34(32)
      acd34(26)=abb34(30)
      acd34(27)=abb34(22)
      acd34(28)=abb34(16)
      acd34(29)=abb34(11)
      acd34(30)=acd34(3)*acd34(1)
      acd34(31)=acd34(8)*acd34(7)
      acd34(32)=-acd34(10)*acd34(9)
      acd34(30)=-acd34(11)+acd34(32)+acd34(31)+acd34(30)
      acd34(30)=acd34(2)*acd34(30)
      acd34(31)=-acd34(5)*acd34(1)
      acd34(32)=acd34(12)*acd34(7)
      acd34(33)=-acd34(13)*acd34(9)
      acd34(31)=-acd34(14)+acd34(33)+acd34(32)+acd34(31)
      acd34(31)=acd34(4)*acd34(31)
      acd34(32)=acd34(19)*acd34(18)
      acd34(33)=acd34(21)*acd34(20)
      acd34(32)=-acd34(22)+acd34(33)+acd34(32)
      acd34(32)=acd34(17)*acd34(32)
      acd34(33)=acd34(24)*acd34(18)
      acd34(34)=acd34(26)*acd34(20)
      acd34(33)=-acd34(28)+acd34(34)+acd34(33)
      acd34(33)=acd34(23)*acd34(33)
      acd34(34)=-acd34(6)*acd34(1)
      acd34(35)=-acd34(15)*acd34(7)
      acd34(36)=-acd34(16)*acd34(9)
      acd34(37)=-acd34(25)*acd34(18)
      acd34(38)=-acd34(27)*acd34(20)
      brack=acd34(29)+acd34(30)+acd34(31)+acd34(32)+acd34(33)+acd34(34)+acd34(3&
      &5)+acd34(36)+acd34(37)+acd34(38)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd34h0
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(48) :: acd34
      complex(ki) :: brack
      acd34(1)=spvak2e1(iv1)
      acd34(2)=dotproduct(qshift,spvae1k2)
      acd34(3)=abb34(13)
      acd34(4)=dotproduct(qshift,spvae1l3)
      acd34(5)=abb34(20)
      acd34(6)=abb34(10)
      acd34(7)=spvae1k2(iv1)
      acd34(8)=dotproduct(qshift,spvak2e1)
      acd34(9)=dotproduct(qshift,spval5e1)
      acd34(10)=abb34(27)
      acd34(11)=dotproduct(qshift,spvae2e1)
      acd34(12)=abb34(23)
      acd34(13)=abb34(25)
      acd34(14)=spvae1l3(iv1)
      acd34(15)=abb34(19)
      acd34(16)=abb34(24)
      acd34(17)=abb34(18)
      acd34(18)=spval5e1(iv1)
      acd34(19)=abb34(36)
      acd34(20)=spvae2e1(iv1)
      acd34(21)=abb34(14)
      acd34(22)=spval3e1(iv1)
      acd34(23)=dotproduct(qshift,spvae1l5)
      acd34(24)=abb34(48)
      acd34(25)=dotproduct(qshift,spvae1e2)
      acd34(26)=abb34(50)
      acd34(27)=abb34(12)
      acd34(28)=spvae1l5(iv1)
      acd34(29)=dotproduct(qshift,spval3e1)
      acd34(30)=dotproduct(qshift,spval4e1)
      acd34(31)=abb34(52)
      acd34(32)=abb34(32)
      acd34(33)=spvae1e2(iv1)
      acd34(34)=abb34(30)
      acd34(35)=abb34(22)
      acd34(36)=spval4e1(iv1)
      acd34(37)=abb34(16)
      acd34(38)=acd34(11)*acd34(16)
      acd34(39)=-acd34(9)*acd34(15)
      acd34(40)=acd34(5)*acd34(8)
      acd34(38)=acd34(40)+acd34(39)+acd34(17)+acd34(38)
      acd34(38)=acd34(14)*acd34(38)
      acd34(39)=acd34(11)*acd34(12)
      acd34(40)=-acd34(9)*acd34(10)
      acd34(41)=-acd34(3)*acd34(8)
      acd34(39)=acd34(41)+acd34(40)+acd34(13)+acd34(39)
      acd34(39)=acd34(7)*acd34(39)
      acd34(40)=-acd34(25)*acd34(34)
      acd34(41)=-acd34(23)*acd34(31)
      acd34(40)=acd34(41)+acd34(37)+acd34(40)
      acd34(40)=acd34(36)*acd34(40)
      acd34(41)=-acd34(30)*acd34(34)
      acd34(42)=-acd34(26)*acd34(29)
      acd34(41)=acd34(42)+acd34(35)+acd34(41)
      acd34(41)=acd34(33)*acd34(41)
      acd34(42)=-acd34(30)*acd34(31)
      acd34(43)=-acd34(24)*acd34(29)
      acd34(42)=acd34(43)+acd34(32)+acd34(42)
      acd34(42)=acd34(28)*acd34(42)
      acd34(43)=-acd34(25)*acd34(26)
      acd34(44)=-acd34(23)*acd34(24)
      acd34(43)=acd34(44)+acd34(27)+acd34(43)
      acd34(43)=acd34(22)*acd34(43)
      acd34(44)=acd34(20)*acd34(16)
      acd34(45)=-acd34(18)*acd34(15)
      acd34(44)=acd34(44)+acd34(45)
      acd34(44)=acd34(4)*acd34(44)
      acd34(45)=acd34(20)*acd34(12)
      acd34(46)=-acd34(18)*acd34(10)
      acd34(45)=acd34(45)+acd34(46)
      acd34(45)=acd34(2)*acd34(45)
      acd34(46)=acd34(4)*acd34(5)
      acd34(47)=-acd34(2)*acd34(3)
      acd34(46)=acd34(47)+acd34(6)+acd34(46)
      acd34(46)=acd34(1)*acd34(46)
      acd34(47)=acd34(20)*acd34(21)
      acd34(48)=acd34(18)*acd34(19)
      brack=acd34(38)+acd34(39)+acd34(40)+acd34(41)+acd34(42)+acd34(43)+acd34(4&
      &4)+acd34(45)+acd34(46)+acd34(47)+acd34(48)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd34h0
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(37) :: acd34
      complex(ki) :: brack
      acd34(1)=spvak2e1(iv1)
      acd34(2)=spvae1k2(iv2)
      acd34(3)=abb34(13)
      acd34(4)=spvae1l3(iv2)
      acd34(5)=abb34(20)
      acd34(6)=spvak2e1(iv2)
      acd34(7)=spvae1k2(iv1)
      acd34(8)=spvae1l3(iv1)
      acd34(9)=spval5e1(iv2)
      acd34(10)=abb34(27)
      acd34(11)=spvae2e1(iv2)
      acd34(12)=abb34(23)
      acd34(13)=spval5e1(iv1)
      acd34(14)=spvae2e1(iv1)
      acd34(15)=abb34(19)
      acd34(16)=abb34(24)
      acd34(17)=spval3e1(iv1)
      acd34(18)=spvae1l5(iv2)
      acd34(19)=abb34(48)
      acd34(20)=spvae1e2(iv2)
      acd34(21)=abb34(50)
      acd34(22)=spval3e1(iv2)
      acd34(23)=spvae1l5(iv1)
      acd34(24)=spvae1e2(iv1)
      acd34(25)=spval4e1(iv2)
      acd34(26)=abb34(52)
      acd34(27)=spval4e1(iv1)
      acd34(28)=abb34(30)
      acd34(29)=-acd34(11)*acd34(16)
      acd34(30)=acd34(9)*acd34(15)
      acd34(31)=-acd34(5)*acd34(6)
      acd34(29)=acd34(31)+acd34(29)+acd34(30)
      acd34(29)=acd34(8)*acd34(29)
      acd34(30)=-acd34(11)*acd34(12)
      acd34(31)=acd34(9)*acd34(10)
      acd34(32)=acd34(3)*acd34(6)
      acd34(30)=acd34(32)+acd34(30)+acd34(31)
      acd34(30)=acd34(7)*acd34(30)
      acd34(31)=-acd34(14)*acd34(16)
      acd34(32)=acd34(13)*acd34(15)
      acd34(33)=-acd34(1)*acd34(5)
      acd34(31)=acd34(33)+acd34(31)+acd34(32)
      acd34(31)=acd34(4)*acd34(31)
      acd34(32)=-acd34(12)*acd34(14)
      acd34(33)=acd34(10)*acd34(13)
      acd34(34)=acd34(1)*acd34(3)
      acd34(32)=acd34(34)+acd34(32)+acd34(33)
      acd34(32)=acd34(2)*acd34(32)
      acd34(33)=acd34(20)*acd34(28)
      acd34(34)=acd34(18)*acd34(26)
      acd34(33)=acd34(34)+acd34(33)
      acd34(33)=acd34(27)*acd34(33)
      acd34(34)=acd34(24)*acd34(28)
      acd34(35)=acd34(23)*acd34(26)
      acd34(34)=acd34(34)+acd34(35)
      acd34(34)=acd34(25)*acd34(34)
      acd34(35)=acd34(21)*acd34(24)
      acd34(36)=acd34(19)*acd34(23)
      acd34(35)=acd34(36)+acd34(35)
      acd34(35)=acd34(22)*acd34(35)
      acd34(36)=acd34(20)*acd34(21)
      acd34(37)=acd34(18)*acd34(19)
      acd34(36)=acd34(36)+acd34(37)
      acd34(36)=acd34(17)*acd34(36)
      brack=acd34(29)+acd34(30)+acd34(31)+acd34(32)+acd34(33)+acd34(34)+acd34(3&
      &5)+acd34(36)
   end function brack_3
!---#] function brack_3:
!---#[ function derivative:
   function derivative(mu2,i1,i2) result(numerator)
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd34h0
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
      qshift = k2-k5
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
end module     p2_gg_httbar_d34h0l1d
