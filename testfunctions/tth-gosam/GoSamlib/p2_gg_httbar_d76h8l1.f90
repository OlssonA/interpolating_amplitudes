module     p2_gg_httbar_d76h8l1
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d76h8l1.f90
   ! generator: buildfortran.py
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_util, only: cond
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   public :: numerator_ninja
contains
!---#[ function brack_1:
   pure function brack_1(Q,mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd76h8
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc76(39)
      complex(ki) :: Qspvae2e1
      complex(ki) :: Qspval4e1
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspvae1l5
      complex(ki) :: Qspvae1l4
      complex(ki) :: Qspvak2e1
      complex(ki) :: QspQ
      complex(ki) :: Qspvae1l3
      complex(ki) :: Qspvae1k2
      complex(ki) :: Qspval3e1
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      Qspval4e1 = dotproduct(Q,spval4e1)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspvae1l5 = dotproduct(Q,spvae1l5)
      Qspvae1l4 = dotproduct(Q,spvae1l4)
      Qspvak2e1 = dotproduct(Q,spvak2e1)
      QspQ = dotproduct(Q,Q)
      Qspvae1l3 = dotproduct(Q,spvae1l3)
      Qspvae1k2 = dotproduct(Q,spvae1k2)
      Qspval3e1 = dotproduct(Q,spval3e1)
      acc76(1)=abb76(9)
      acc76(2)=abb76(10)
      acc76(3)=abb76(11)
      acc76(4)=abb76(12)
      acc76(5)=abb76(13)
      acc76(6)=abb76(14)
      acc76(7)=abb76(15)
      acc76(8)=abb76(16)
      acc76(9)=abb76(17)
      acc76(10)=abb76(18)
      acc76(11)=abb76(19)
      acc76(12)=abb76(20)
      acc76(13)=abb76(21)
      acc76(14)=abb76(22)
      acc76(15)=abb76(23)
      acc76(16)=abb76(24)
      acc76(17)=abb76(25)
      acc76(18)=abb76(26)
      acc76(19)=abb76(27)
      acc76(20)=abb76(28)
      acc76(21)=abb76(29)
      acc76(22)=abb76(33)
      acc76(23)=abb76(34)
      acc76(24)=abb76(35)
      acc76(25)=abb76(38)
      acc76(26)=abb76(39)
      acc76(27)=abb76(40)
      acc76(28)=abb76(41)
      acc76(29)=abb76(42)
      acc76(30)=acc76(5)*Qspvae2e1
      acc76(31)=-acc76(12)*Qspval4e1
      acc76(32)=-acc76(14)*Qspvae1e2
      acc76(33)=acc76(16)*Qspvae1l5
      acc76(34)=-acc76(17)*Qspvae1l4
      acc76(35)=acc76(20)*Qspvak2e1
      acc76(30)=acc76(35)+acc76(34)+acc76(33)+acc76(32)+acc76(31)+acc76(8)+acc7&
      &6(30)
      acc76(30)=QspQ*acc76(30)
      acc76(31)=acc76(2)*Qspvae1l4
      acc76(32)=acc76(4)*Qspvae1l5
      acc76(33)=acc76(7)*Qspvae1e2
      acc76(34)=acc76(19)*Qspvae1l3
      acc76(35)=acc76(21)*Qspvae1k2
      acc76(31)=acc76(35)+acc76(34)+acc76(10)+acc76(33)+acc76(32)+acc76(31)
      acc76(31)=Qspvak2e1*acc76(31)
      acc76(32)=-acc76(25)*Qspvae1l5
      acc76(33)=acc76(27)*Qspvae1e2
      acc76(34)=acc76(28)*Qspvae1l4
      acc76(32)=acc76(34)+acc76(33)+acc76(26)+acc76(32)
      acc76(32)=Qspval3e1*acc76(32)
      acc76(33)=acc76(18)*Qspvae2e1
      acc76(34)=acc76(23)*Qspval4e1
      acc76(33)=acc76(34)+acc76(33)+acc76(6)
      acc76(33)=Qspvae1l5*acc76(33)
      acc76(34)=-acc76(11)*Qspvae2e1
      acc76(35)=acc76(24)*Qspval4e1
      acc76(34)=acc76(35)+acc76(13)+acc76(34)
      acc76(34)=Qspvae1l3*acc76(34)
      acc76(35)=acc76(3)*Qspvae1l4
      acc76(36)=acc76(9)*Qspvae1e2
      acc76(37)=acc76(15)*Qspvae2e1
      acc76(38)=acc76(22)*Qspval4e1
      acc76(39)=acc76(29)*Qspvae1k2
      brack=acc76(1)+acc76(30)+acc76(31)+acc76(32)+acc76(33)+acc76(34)+acc76(35&
      &)+acc76(36)+acc76(37)+acc76(38)+acc76(39)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d76h8l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd76h8
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d76
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = k3+k5
      Q(1:4)  =cmplx(real(-Q_ext(0:3)  -qshift(:),  ki_nin), aimag(-Q_ext(0:3))&
      &, ki)
      d76 = 0.0_ki
      d76 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d76, ki), aimag(d76), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d76h8l1
