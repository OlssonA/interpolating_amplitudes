module     p2_gg_httbar_d162h8l1d
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d162h8l1d.f90
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
      use p2_gg_httbar_abbrevd162h8
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(40) :: acd162
      complex(ki) :: brack
      acd162(1)=dotproduct(qshift,spvak1e1)
      acd162(2)=dotproduct(qshift,spvae1l5)
      acd162(3)=abb162(21)
      acd162(4)=abb162(16)
      acd162(5)=dotproduct(qshift,spvak2e1)
      acd162(6)=abb162(30)
      acd162(7)=dotproduct(qshift,spval4e1)
      acd162(8)=abb162(39)
      acd162(9)=dotproduct(qshift,spval5e1)
      acd162(10)=abb162(33)
      acd162(11)=dotproduct(qshift,spvae2e1)
      acd162(12)=abb162(17)
      acd162(13)=abb162(14)
      acd162(14)=dotproduct(qshift,spvae1k1)
      acd162(15)=abb162(13)
      acd162(16)=abb162(18)
      acd162(17)=dotproduct(qshift,spvae1k2)
      acd162(18)=abb162(26)
      acd162(19)=dotproduct(qshift,spvae1l4)
      acd162(20)=abb162(43)
      acd162(21)=dotproduct(qshift,spvae1e2)
      acd162(22)=abb162(23)
      acd162(23)=abb162(15)
      acd162(24)=abb162(25)
      acd162(25)=abb162(34)
      acd162(26)=abb162(19)
      acd162(27)=abb162(35)
      acd162(28)=abb162(29)
      acd162(29)=abb162(20)
      acd162(30)=abb162(12)
      acd162(31)=acd162(6)*acd162(2)
      acd162(32)=acd162(15)*acd162(14)
      acd162(33)=acd162(18)*acd162(17)
      acd162(34)=-acd162(20)*acd162(19)
      acd162(35)=-acd162(22)*acd162(21)
      acd162(31)=-acd162(23)+acd162(35)+acd162(34)+acd162(33)+acd162(32)+acd162&
      &(31)
      acd162(31)=acd162(5)*acd162(31)
      acd162(32)=acd162(3)*acd162(1)
      acd162(33)=acd162(8)*acd162(7)
      acd162(34)=acd162(10)*acd162(9)
      acd162(35)=acd162(12)*acd162(11)
      acd162(32)=-acd162(13)+acd162(35)+acd162(34)+acd162(33)+acd162(32)
      acd162(32)=acd162(2)*acd162(32)
      acd162(33)=-acd162(4)*acd162(1)
      acd162(34)=-acd162(16)*acd162(14)
      acd162(35)=-acd162(24)*acd162(17)
      acd162(36)=-acd162(25)*acd162(19)
      acd162(37)=-acd162(26)*acd162(21)
      acd162(38)=-acd162(27)*acd162(7)
      acd162(39)=-acd162(28)*acd162(9)
      acd162(40)=-acd162(29)*acd162(11)
      brack=acd162(30)+acd162(31)+acd162(32)+acd162(33)+acd162(34)+acd162(35)+a&
      &cd162(36)+acd162(37)+acd162(38)+acd162(39)+acd162(40)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd162h8
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(51) :: acd162
      complex(ki) :: brack
      acd162(1)=spvak1e1(iv1)
      acd162(2)=dotproduct(qshift,spvae1l5)
      acd162(3)=abb162(21)
      acd162(4)=abb162(16)
      acd162(5)=spvae1l5(iv1)
      acd162(6)=dotproduct(qshift,spvak1e1)
      acd162(7)=dotproduct(qshift,spvak2e1)
      acd162(8)=abb162(30)
      acd162(9)=dotproduct(qshift,spval4e1)
      acd162(10)=abb162(39)
      acd162(11)=dotproduct(qshift,spval5e1)
      acd162(12)=abb162(33)
      acd162(13)=dotproduct(qshift,spvae2e1)
      acd162(14)=abb162(17)
      acd162(15)=abb162(14)
      acd162(16)=spvae1k1(iv1)
      acd162(17)=abb162(13)
      acd162(18)=abb162(18)
      acd162(19)=spvak2e1(iv1)
      acd162(20)=dotproduct(qshift,spvae1k1)
      acd162(21)=dotproduct(qshift,spvae1k2)
      acd162(22)=abb162(26)
      acd162(23)=dotproduct(qshift,spvae1l4)
      acd162(24)=abb162(43)
      acd162(25)=dotproduct(qshift,spvae1e2)
      acd162(26)=abb162(23)
      acd162(27)=abb162(15)
      acd162(28)=spvae1k2(iv1)
      acd162(29)=abb162(25)
      acd162(30)=spvae1l4(iv1)
      acd162(31)=abb162(34)
      acd162(32)=spvae1e2(iv1)
      acd162(33)=abb162(19)
      acd162(34)=spval4e1(iv1)
      acd162(35)=abb162(35)
      acd162(36)=spval5e1(iv1)
      acd162(37)=abb162(29)
      acd162(38)=spvae2e1(iv1)
      acd162(39)=abb162(20)
      acd162(40)=acd162(26)*acd162(25)
      acd162(41)=acd162(24)*acd162(23)
      acd162(42)=-acd162(22)*acd162(21)
      acd162(43)=-acd162(17)*acd162(20)
      acd162(44)=-acd162(2)*acd162(8)
      acd162(40)=acd162(44)+acd162(43)+acd162(42)+acd162(41)+acd162(27)+acd162(&
      &40)
      acd162(40)=acd162(19)*acd162(40)
      acd162(41)=-acd162(14)*acd162(13)
      acd162(42)=-acd162(12)*acd162(11)
      acd162(43)=-acd162(10)*acd162(9)
      acd162(44)=-acd162(3)*acd162(6)
      acd162(45)=-acd162(7)*acd162(8)
      acd162(41)=acd162(45)+acd162(44)+acd162(43)+acd162(42)+acd162(15)+acd162(&
      &41)
      acd162(41)=acd162(5)*acd162(41)
      acd162(42)=acd162(26)*acd162(32)
      acd162(43)=acd162(24)*acd162(30)
      acd162(44)=-acd162(22)*acd162(28)
      acd162(45)=-acd162(16)*acd162(17)
      acd162(42)=acd162(45)+acd162(44)+acd162(42)+acd162(43)
      acd162(42)=acd162(7)*acd162(42)
      acd162(43)=-acd162(14)*acd162(38)
      acd162(44)=-acd162(12)*acd162(36)
      acd162(45)=-acd162(10)*acd162(34)
      acd162(46)=-acd162(1)*acd162(3)
      acd162(43)=acd162(46)+acd162(45)+acd162(43)+acd162(44)
      acd162(43)=acd162(2)*acd162(43)
      acd162(44)=acd162(38)*acd162(39)
      acd162(45)=acd162(36)*acd162(37)
      acd162(46)=acd162(34)*acd162(35)
      acd162(47)=acd162(32)*acd162(33)
      acd162(48)=acd162(30)*acd162(31)
      acd162(49)=acd162(28)*acd162(29)
      acd162(50)=acd162(16)*acd162(18)
      acd162(51)=acd162(1)*acd162(4)
      brack=acd162(40)+acd162(41)+acd162(42)+acd162(43)+acd162(44)+acd162(45)+a&
      &cd162(46)+acd162(47)+acd162(48)+acd162(49)+acd162(50)+acd162(51)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd162h8
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(36) :: acd162
      complex(ki) :: brack
      acd162(1)=spvak1e1(iv1)
      acd162(2)=spvae1l5(iv2)
      acd162(3)=abb162(21)
      acd162(4)=spvak1e1(iv2)
      acd162(5)=spvae1l5(iv1)
      acd162(6)=spvak2e1(iv2)
      acd162(7)=abb162(30)
      acd162(8)=spval4e1(iv2)
      acd162(9)=abb162(39)
      acd162(10)=spval5e1(iv2)
      acd162(11)=abb162(33)
      acd162(12)=spvae2e1(iv2)
      acd162(13)=abb162(17)
      acd162(14)=spvak2e1(iv1)
      acd162(15)=spval4e1(iv1)
      acd162(16)=spval5e1(iv1)
      acd162(17)=spvae2e1(iv1)
      acd162(18)=spvae1k1(iv1)
      acd162(19)=abb162(13)
      acd162(20)=spvae1k1(iv2)
      acd162(21)=spvae1k2(iv2)
      acd162(22)=abb162(26)
      acd162(23)=spvae1l4(iv2)
      acd162(24)=abb162(43)
      acd162(25)=spvae1e2(iv2)
      acd162(26)=abb162(23)
      acd162(27)=spvae1k2(iv1)
      acd162(28)=spvae1l4(iv1)
      acd162(29)=spvae1e2(iv1)
      acd162(30)=acd162(13)*acd162(12)
      acd162(31)=acd162(11)*acd162(10)
      acd162(32)=acd162(9)*acd162(8)
      acd162(33)=acd162(3)*acd162(4)
      acd162(34)=acd162(6)*acd162(7)
      acd162(30)=acd162(34)+acd162(33)+acd162(32)+acd162(30)+acd162(31)
      acd162(30)=acd162(5)*acd162(30)
      acd162(31)=acd162(13)*acd162(17)
      acd162(32)=acd162(11)*acd162(16)
      acd162(33)=acd162(9)*acd162(15)
      acd162(34)=acd162(3)*acd162(1)
      acd162(35)=acd162(14)*acd162(7)
      acd162(31)=acd162(35)+acd162(34)+acd162(33)+acd162(31)+acd162(32)
      acd162(31)=acd162(2)*acd162(31)
      acd162(32)=-acd162(26)*acd162(25)
      acd162(33)=-acd162(24)*acd162(23)
      acd162(34)=acd162(22)*acd162(21)
      acd162(35)=acd162(19)*acd162(20)
      acd162(32)=acd162(35)+acd162(34)+acd162(32)+acd162(33)
      acd162(32)=acd162(14)*acd162(32)
      acd162(33)=-acd162(26)*acd162(29)
      acd162(34)=-acd162(24)*acd162(28)
      acd162(35)=acd162(22)*acd162(27)
      acd162(36)=acd162(19)*acd162(18)
      acd162(33)=acd162(36)+acd162(35)+acd162(33)+acd162(34)
      acd162(33)=acd162(6)*acd162(33)
      brack=acd162(30)+acd162(31)+acd162(32)+acd162(33)
   end function brack_3
!---#] function brack_3:
!---#[ function derivative:
   function derivative(mu2,i1,i2) result(numerator)
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd162h8
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
end module     p2_gg_httbar_d162h8l1d
